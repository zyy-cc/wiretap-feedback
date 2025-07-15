import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
import copy
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm
import argparse

import math


from torch.autograd import Variable

from mine.models.layers import ConcatLayer, CustomSequential
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import mine.utils

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6
device = 'cuda:0'

############################################################
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in utils.batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
                #print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        return final_mi


class T(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)


class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, x_dim, z_dim, loss='mine', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.T = T(x_dim, z_dim)
        
        self.energy_loss = Mine(self.T, loss=loss, alpha=kwargs['alpha'])

        self.kwargs = kwargs

        self.train_loader = kwargs.get('train_loader')
        self.test_loader = kwargs.get('test_loader')
        self.avg_test_mi = None
        self.test_outputs = []

    def forward(self, x, z):
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        return self.energy_loss(x, z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])

    def training_step(self, batch, batch_idx):

        x, z = batch

        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        loss = self.energy_loss(x, z)
        mi = -loss
        tensorboard_logs = {'loss': loss, 'mi': mi}
        tqdm_dict = {'loss_tqdm': loss, 'mi': mi}

        return {
            **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        x, z = batch
        loss = self.energy_loss(x, z)
        mi = -loss
        output = {'test_loss': loss, 'test_mi': mi}
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        test_mi_values = torch.stack([x['test_mi'] for x in self.test_outputs])
        avg_test_mi = test_mi_values.mean().item()
        self.avg_test_mi = avg_test_mi
        self.log('avg_test_mi', avg_test_mi)
        self.test_outputs.clear()
        # tensorboard_logs = {'test_mi': avg_mi}
        # return {'avg_test_mi': avg_mi, 'log': tensorboard_logs}


    def train_dataloader(self):
        if self.train_loader:
            return self.train_loader

        train_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return train_loader

    def test_dataloader(self):
        if self.test_loader:
            return self.train_loader

        test_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return test_loader

def bin2dec(binary_data, k):
    """
    Transform binary message bits to real value.
    Input: (num_samples, 1, k)
    Output: (num_samples, 1, 1)
    """
    power = (2 ** torch.arange(k - 1, -1, -1, device=binary_data.device, dtype=binary_data.dtype)).float()
    decimal_output = torch.matmul(binary_data.float(), power.unsqueeze(-1))  # Shape: (num_samples, 1, 1)
    
    return decimal_output

def dec2bin(decimal_data, k):
    """
    Transform real value to message bits.
    Input: (num_samples, 1, 1)
    Output: (num_samples, 1, k)
    """
    power = (2 ** torch.arange(k - 1, -1, -1, device=decimal_data.device)).long()
    boolean = torch.bitwise_and(decimal_data.long(), power.view(1, 1, -1)) > 0
    binary_output = boolean.to(dtype=torch.int64)  # Convert boolean to int
    
    return binary_output


def gf2_exp_log_tables(q, prim_poly):
    """ Generates exponent and log tables for GF(2^q) using the given primitive polynomial. """
    field_size = 2 ** q
    exp_table = torch.zeros(field_size, dtype=torch.int64)
    log_table = torch.full((field_size,), -1, dtype=torch.int64)  # Use -1 for undefined log(0)
    
    x = 1 
    for i in range(field_size - 1):  
        exp_table[i] = x
        log_table[x] = i  
        x <<= 1  # Multiply by alpha (which is x in GF(2)) left shift by 1
        if x & field_size:  # If x exceeds 2^q, apply modulo reduction
            x ^= prim_poly
    
    exp_table[field_size - 1] = 1  # Wrap around for exponentiation
    
    return exp_table, log_table



def gf2_mul(a, b, exp_table, log_table, q, device):
    """
    Performs element-wise multiplication in GF(2^q) using log/exp tables.
    Input: a (batch_size, 1, q), b (batch_size, 1, q)
    Output: res (batch_size, 1, q)
    """
    field_size = 2 ** q
    res = torch.zeros_like(a).to(device)

    mask = (a != 0) & (b != 0)  
    if torch.any(mask):  
        a_flat = a[mask].long()  # Extracts valid elements (1D tensor)
        b_flat = b[mask].long() 

        log_a = log_table[a_flat]
        log_b = log_table[b_flat]

        res[mask] = exp_table[(log_a + log_b) % (field_size - 1)]
    
    return res

def gf2_inv(a, exp_table, log_table, q, device):
    """Computes the multiplicative inverse of elements in GF(2^q)."""
    field_size = 2 ** q
    res = torch.zeros_like(a).to(device)

    mask = a != 0  # Avoid division by zero
    if torch.any(mask):  
        a_flat = a[mask]  
        log_a = log_table[a_flat]  
        inv_values = exp_table[(field_size - 1 - log_a) % (field_size - 1)]  
        res[mask] = inv_values  
    return res

def get_table(q):
    """
    assume q = 4
    """
    
    if q == 3:
        prim_poly = 0b1011  # x^3 + x + 1
    elif q == 4:
        prim_poly = 0b10011  # x^4 + x + 1
    elif q == 5:
        prim_poly = 0b100101  # x^5 + x^2 + 1
    else:
        raise ValueError(f"No primitive polynomial defined for q={q}")

    exp_table, log_table = gf2_exp_log_tables(q, prim_poly)
    exp_table, log_table = exp_table.to(device), log_table.to(device)
    return exp_table, log_table

def secrecy_encode(q, combinedVec, s, exp_table, log_table, device):
    """ 
    Implements efficient secrecy encoding s^{-1} * combinedVec in GF(2^q).
    Output is binary
    """
    batch_size = combinedVec.shape[0]
    decimal_combinedVec = bin2dec(combinedVec, q) # convert to decimal integer
    
    # Convert s from bit-list to GF(2^q) integer
    s_tensor = torch.tensor([int(''.join(map(str, s)), 2)], dtype=torch.int64, device=device)
    s_inverse = gf2_inv(s_tensor, exp_table, log_table, q, device) # gets the vector integer
    s_inverse = s_inverse.view(1, 1, 1).expand(batch_size, 1, 1)
    result_decimal = gf2_mul(s_inverse.to(device), decimal_combinedVec.to(device), exp_table, log_table, q, device)
    result_binary = dec2bin(result_decimal, q)
    return s_tensor, result_decimal, result_binary

def secrecy_decode(q, combinedVec_estimate_decimal, s_tensor, k, exp_table, log_table):
    """ 
    Implements efficient secrecy encoding s * combinedVec_estimate in GF(2^q), taking k most significant bits.
    Input: combinedVec_estimate (batch_size, 1, 1) decimal
    Output: (s * combinedVec_estimate)k which selects k most significant bits, binary
    """
    batch_size = combinedVec_estimate_decimal.shape[0]
    device = combinedVec_estimate_decimal.device
    s_tensor = s_tensor.view(1, 1, 1).expand(batch_size, 1, 1)
    dec_output = gf2_mul(s_tensor, combinedVec_estimate_decimal, exp_table, log_table, q, device)
    binary_output = dec2bin(dec_output, q)
    return binary_output[:, :, :k]



# SK 
def get_args(jupyter_notebook):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-power', type = int, default = 1)

    # message bits
    parser.add_argument('-K', type = int, default=3) 
    parser.add_argument('-m', type = int, default=3) 
    parser.add_argument('-N', type = int, default=9)  
    parser.add_argument('-T', type = int, default=9)  
    parser.add_argument('-q', type = int, default=4)  
    parser.add_argument('-ell', type = int, default=1)  
   

    # channel definition
    parser.add_argument('--snr1_bob', type=int, default= 0, help="Transmission SNR for Bob")
    parser.add_argument('--snr1_eve', type=int, default= 0, help="Transmission SNR for Eve")
    parser.add_argument('--snr2_bob', type=int, default= 100, help="Feedback SNR for Bob")
    parser.add_argument('--snr2_eve', type=int, default= 100, help="Feedback SNR for Eve")
    parser.add_argument('-device', type=str, default='cuda:0', help="GPU")
    parser.add_argument('-num_samples', type=int, default=100000) 
    if jupyter_notebook:
        args = parser.parse_args(args=[])   # for jupyter notebook
    else:
        args = parser.parse_args()    # in general
    return args

def PAMmodulation(binary_data, k):
    """
    Input: m (num_samples, k)
    Output: theta (num_samples, 1)
    """

    M = 2**k
    decimal_data = bin2dec(binary_data, k)
    eta = torch.sqrt(torch.tensor(3)/(M**2 -1))
    theta = (2 * decimal_data - (M - 1)) * eta 

    return decimal_data, theta

def PAMdemodulation(noisy_theta, k):
    """
    Input: noisy theta (num_samples, 1)
    Output: message bits (num_samples, k)
    """
    M = 2**k
    device = noisy_theta.device
    eta = torch.sqrt(torch.tensor(3)/(M**2 -1))
    min_val = (-(M - 1) * eta).clone().detach().to(device)
    max_val = ((M - 1) * eta).clone().detach().to(device)
    noisy_theta_clamp = torch.clamp(noisy_theta, min=min_val, max=max_val)
    decimal_data = torch.round((noisy_theta_clamp/eta + M-1)/2).to(dtype=torch.int64)
    decoding_output = dec2bin(decimal_data, k)
    return decimal_data, decoding_output


def normalize(theta, P):
    """
    normalize data to satisfy power constraint P
    Input: theta (num_samples, 1)
    Output: theta (num_samples, 1)
    """
    # normalize the data based on data
    theta_mean = torch.mean(theta, 0)
    theta_std = torch.std(theta,0)
    normalized_theta = torch.sqrt(P)  * ((theta - theta_mean)*1.0/theta_std) 
    return normalized_theta


def snr_db_2_sigma(snr_db, feedback=False):
    if feedback and snr_db == 100:
        return 0
    return 10**(-snr_db * 1.0 / 20)

def generate_data(K, N, forward_SNR, feedback_SNR, num_samples):
    """
    K: length of message bits
    N: number of rounds
    Output: 
    message (num_samples, K)
    forward_noise (num_samples, N)
    feedback_noise (num_samples, N)
    """
    message = torch.randint(0, 2, (num_samples, K))
    forward_sigma = snr_db_2_sigma(forward_SNR)
    feedback_sigma = snr_db_2_sigma(feedback_SNR, feedback=True)
    forward_noise = forward_sigma * torch.randn((num_samples, N))
    feedback_noise = feedback_sigma * torch.randn((num_samples, N))
    return message, forward_noise, feedback_noise



def sk(message1, forward_noise1, feedback_noise1, forward_noise2, feedback_noise2, args):
    num_samples = message1.shape[0]
    P = torch.tensor(args.power).to(args.device)

    sigma1 = snr_db_2_sigma(args.snr1_bob)

    # encoding 
    message1_index, theta1 = PAMmodulation(message1, args.q)
    
    # first transmission
    X1 = torch.sqrt(P) * theta1
    
    # print(f"transmission: {1}, power: {torch.var(X1)}")
    # print(f"transmission: {2}, power: {torch.var(X2)}")
    
    Y1 = X1 + forward_noise1[:,:, 0].view(X1.shape[0], args.ell, 1) 
    Z1 = X1 + forward_noise2[:,:, 0].view(X1.shape[0], args.ell, 1)
    received_bob = Y1
    received_eve = Z1
    for t in range(1, args.N + 1):
        if t == 1:
            # LMMSE estimator
            theta1_estimate = torch.sqrt(P) * Y1 /(P + sigma1**2)
            
            epsilon1 = theta1_estimate - theta1
            alpha1 = torch.var(epsilon1)
            
            X_total = X1
        
        else:
            Xt = epsilon1 * torch.sqrt(P)/torch.sqrt(alpha1) 
            
            X_total = torch.cat([X_total, Xt], dim = 1)
            # print(f"transmission: {t}, power: {torch.var(Xt)}")
            Y1t = Xt + forward_noise1[:,:, t - 1].view(X1.shape[0], args.ell, 1)
            Z1t = Xt + forward_noise2[:,:, t - 1].view(X1.shape[0], args.ell, 1)

            received_bob = torch.cat([received_bob, Y1t], dim = 1)
            received_eve = torch.cat([received_eve, Z1t], dim = 1)

            coe1 = torch.sqrt(P) * torch.sqrt(alpha1) /(P + sigma1**2)

            theta1_estimate = theta1_estimate - Y1t * coe1
            
            # estimation error
            epsilon1 = theta1_estimate - theta1
        
            alpha1 = torch.var(epsilon1)
        

    # decoding 
    message1_index_pred, message1_pred = PAMdemodulation(theta1_estimate, args.q)

    bler1 = calculate_bler(message1, message1_pred)
    
    return message1_index_pred, message1_pred, bler1, received_bob, received_eve, X_total



def calculate_bler(tensor1, tensor2):
    """
    Calculate Block Error Rate (BLER) for two tensors of shape (num_samples, 1, k).
    
    Args:
        tensor1 (torch.Tensor): Ground truth tensor, shape (num_samples, 1, k).
        tensor2 (torch.Tensor): Predicted tensor, shape (num_samples, 1, k).
    
    Returns:
        bler (float): Block Error Rate.
    """
    # Ensure shapes match
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    
    correct_blocks = torch.all(tensor1 == tensor2, dim=2)  # Shape: (num_samples, 1)
    incorrect_blocks = (~correct_blocks).sum().item()  # Sum over all dimensions
    
    # Calculate BLER
    total_blocks = tensor1.shape[0] 
    bler = incorrect_blocks / total_blocks
    
    return bler

args = get_args(False)
print('args = ', args.__dict__)

exp_table, log_table = get_table(args.q)
s = [1, 1, 0, 1]

args.numTest = 10000000

mVec_1 = torch.randint(0, 2, (args.numTest, args.ell, args.m))
bVec_1 = torch.randint(0, 2, (args.numTest, args.ell, args.q - args.m))

combedVec_1 = torch.cat([mVec_1, bVec_1], dim = 2)
s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, s, exp_table, log_table, args.device)
std1_bob = 10 ** (-args.snr1_bob * 1.0 / 10 / 2) #forward snr
std1_eve = 10 ** (-args.snr1_eve * 1.0 / 10 / 2) #forward snr

std2_bob = 10 ** (-args.snr2_bob * 1.0 / 10 / 2) #feedback snr
std2_eve = 10 ** (-args.snr2_eve * 1.0 / 10 / 2) #feedback snr

# Noise values for the parity bits
fwd_noise_bob = torch.normal(0, std=std1_bob, size=(args.numTest, args.ell, args.T), requires_grad=False)
fb_noise_bob = torch.normal(0, std=std2_bob, size=(args.numTest, args.ell, args.T), requires_grad=False)
fwd_noise_eve = torch.normal(0, std=std1_eve, size=(args.numTest, args.ell, args.T), requires_grad=False)
fb_noise_eve = torch.normal(0, std=std2_eve, size=(args.numTest, args.ell, args.T), requires_grad=False)

if args.snr2_bob >= 100:
    fb_noise_bob = 0* fb_noise_bob
if args.snr2_eve >= 100:
    fb_noise_eve = 0* fb_noise_eve

    
message1_index_pred, message1_pred, bler1, received_bob, received_eve, X_total = sk(secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), args)


secre_decode_input = message1_index_pred.reshape(-1, 1, 1)
secre_decode_output = secrecy_decode(args.q, secre_decode_input, s_tensor, args.m, exp_table, log_table)

secry_bler = calculate_bler(mVec_1.to(device), secre_decode_output)
print('the BLER for Bob is:', bler1)
print('after secrey the BLER for Bob is:', secry_bler)
print("total power", torch.mean(torch.square(X_total)))

def estimate_mi(s, args):
    exp_table, log_table = get_table(args.q)
    args.numTest = 1000000

    eve_list = [args.snr1_eve]
    for snr1_eve in eve_list:
        print(f"---------eve {snr1_eve}--------------------")

        mVec_1 = torch.randint(0, 2, (args.numTest, args.ell, args.m))
        bVec_1 = torch.randint(0, 2, (args.numTest, args.ell, args.q - args.m))

        combedVec_1 = torch.cat([mVec_1, bVec_1], dim = 2)
        s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, s, exp_table, log_table, args.device)
        std1_bob = 10 ** (-args.snr1_bob * 1.0 / 10 / 2) #forward snr
        std1_eve = 10 ** (-snr1_eve * 1.0 / 10 / 2) #forward snr
        
        std2_bob = 10 ** (-args.snr2_bob * 1.0 / 10 / 2) #feedback snr
        std2_eve = 10 ** (-args.snr2_eve * 1.0 / 10 / 2) #feedback snr

        # Noise values for the parity bits
        fwd_noise_bob = torch.normal(0, std=std1_bob, size=(args.numTest, args.ell, args.T), requires_grad=False)
        fb_noise_bob = torch.normal(0, std=std2_bob, size=(args.numTest, args.ell, args.T), requires_grad=False)
        fwd_noise_eve = torch.normal(0, std=std1_eve, size=(args.numTest, args.ell, args.T), requires_grad=False)
        fb_noise_eve = torch.normal(0, std=std2_eve, size=(args.numTest, args.ell, args.T), requires_grad=False)

        if args.snr2_bob >= 100:
            fb_noise_bob = 0* fb_noise_bob
        if args.snr2_eve >= 100:
            fb_noise_eve = 0* fb_noise_eve

    
        message1_index_pred, message1_pred, bler1, received_bob, received_eve, X_total = sk(secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), args)
        input_dim = args.K
    
        output_dim = args.N
        N = args.numTest
        epochs1 = 30
        epochs2 = 30
        lr = 1e-3
        batch_size = 5000

        message = mVec_1.reshape(args.numTest, -1).to(args.device)
        bob = received_bob.reshape(args.numTest, -1).to(args.device)
        eve = received_eve.reshape(args.numTest, -1).to(args.device)

        dataset_eve = TensorDataset(message, eve)

        dataset_bob = TensorDataset(message, bob)

        train_loader = DataLoader(dataset_eve, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset_eve, batch_size=batch_size, shuffle=False)

        kwargs = {
        'loss': 'mine',  
        'alpha': 0.1,  
        'lr': lr,  
        'batch_size': batch_size,
        'train_loader': train_loader,
        'test_loader': test_loader
        }

        mi_estimator = MutualInformationEstimator(input_dim, output_dim, **kwargs).to(device)
        trainer = pl.Trainer(max_epochs=epochs1, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
        trainer.fit(mi_estimator)
        trainer.test()

        print("-------------Mutual information about Eve-------------")
        if hasattr(mi_estimator, 'avg_test_mi'):
            print(f"secrecy_seed{s}, bob snr: {args.snr1_bob}, eve snr: {snr1_eve}, Eve Estimated Mutual Information: {mi_estimator.avg_test_mi} nats, {mi_estimator.avg_test_mi / math.log(2)} bits")
        else:
            print("Error: 'avg_test_mi' was not properly set. Check `on_test_epoch_end`.")

        ##bob

        train_loader_bob = DataLoader(dataset_bob, batch_size=batch_size, shuffle=True)
        test_loader_bob = DataLoader(dataset_bob, batch_size=batch_size, shuffle=False)

        kwargs = {
        'loss': 'mine',  
        'alpha': 0.1,  
        'lr': lr,  
        'batch_size': batch_size,
        'train_loader': train_loader_bob,
        'test_loader': test_loader_bob
        }

        mi_estimator_bob = MutualInformationEstimator(input_dim, output_dim, **kwargs).to(device)
        trainer = pl.Trainer(max_epochs=epochs2, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
        trainer.fit(mi_estimator_bob)
        trainer.test()

        print("-------------Mutual information about Bob-------------")
        if hasattr(mi_estimator_bob, 'avg_test_mi'):
            print(f"secrecy_seed{s}, bob snr: {args.snr1_bob}, eve snr: {args.snr1_eve}, Bob Estimated Mutual Information: {mi_estimator_bob.avg_test_mi} nats, {mi_estimator_bob.avg_test_mi / math.log(2)} bits")
        else:
            print("Error: 'avg_test_mi' was not properly set. Check `on_test_epoch_end`.")


estimate_mi(s, args)
