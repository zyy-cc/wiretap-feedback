# WTC-Lightcode channel with CCE loss + information leakage
import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from Feature_extractors_wire import FE
import copy
from parameters_mutual import *
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

######################### MINE ###################################
import math

from torch.autograd import Variable

from mine.models.layers import ConcatLayer, CustomSequential
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import mine.utils

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.kwargs['gamma'])
        return [optimizer], [scheduler]

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
######################### MINE ###################################


def errors_bler(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    # Reshape to (batch_size, k)
    y_true = y_true.view(y_true.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)

    # Compute block errors: 1 if any bit in the block is incorrect
    block_errors = torch.any(y_true != y_pred, dim=1).float()

    bler_err_rate = block_errors.mean().item()  

    return bler_err_rate

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
    eta = torch.sqrt(torch.tensor(3)/(M**2 -1))
    noisy_theta_clamp = torch.clamp(noisy_theta, min = -(M-1)*eta, max = (M-1)*eta)
    decimal_data = torch.round((noisy_theta_clamp/eta + M-1)/2).to(dtype=torch.int64)
    decoding_output = dec2bin(decimal_data, k)
    return decimal_data, decoding_output


def estimate_mi(secrecy_seed, model, train_mean, train_std, args, logging):
    exp_table, log_table = get_table(args.q)

    s = secrecy_seed
       
    snr1_eve_list = [args.snr1_eve]
    for snr1_eve in snr1_eve_list:
        print(f"---------snr eve {snr1_eve}--------------------")
        if args.train == 0:
            path = f'{weights_folder}/model_weights{args.totalbatch-10}.pt'
        
            print(f"Using model from {path}")
            logging.info(f"Using model from {path}")
        
            checkpoint = torch.load(path,map_location=args.device)
        
            # ======================================================= load weights
            model.load_state_dict(checkpoint)
            model = model.to(args.device)
        model.eval()
        map_vec = 2**(torch.arange(args.m))

        args.numTest = 1000000

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

        with torch.no_grad():
            preds1, received_bob, received_eve, _, _, parity_all = model(None, None, secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), isTraining=1)
        
        input_dim = args.K
        output_dim = args.T
        N = args.numTest
        epochs1 = 30
        epochs2 = 30
        lr = 0.001
        batch_size = 5000

        message = mVec_1.reshape(args.numTest, -1).to(args.device)
        codewords = parity_all.reshape(args.numTest, -1).to(args.device)
        bob = received_bob.reshape(args.numTest, -1).to(args.device)
        eve = received_eve.reshape(args.numTest, -1).to(args.device)

        dataset_eve = TensorDataset(message, eve)
        dataset_bob = TensorDataset(message, bob)

        train_loader = DataLoader(dataset_eve, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset_eve, batch_size=batch_size, shuffle=False)

        kwargs = {
        'loss': 'mine',  
        'alpha': 0.01, 
        'gamma':0.98, 
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
            print(f"secrecy_seed{secrecy_seed}, bob snr: {args.snr1_bob}, eve snr: {snr1_eve}, Eve Estimated Mutual Information: {mi_estimator.avg_test_mi} nats, {mi_estimator.avg_test_mi / math.log(2)} bits")
        else:
            print("Error: 'avg_test_mi' was not properly set. Check `on_test_epoch_end`.")

        ##bob

        train_loader_bob = DataLoader(dataset_bob, batch_size=batch_size, shuffle=True)
        test_loader_bob = DataLoader(dataset_bob, batch_size=batch_size, shuffle=False)

        kwargs = {
        'loss': 'mine',  
        'alpha': 0.01, 
        'gamma':0.98, 
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
            print(f"secrecy_seed{secrecy_seed}, bob snr: {args.snr1_bob}, eve snr: {snr1_eve}, Bob Estimated Mutual Information: {mi_estimator_bob.avg_test_mi} nats, {mi_estimator_bob.avg_test_mi / math.log(2)} bits")
        else:
            print("Error: 'avg_test_mi' was not properly set. Check `on_test_epoch_end`.")

################################### reliability layer ###################################
def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class ae_backbone(nn.Module):
    def __init__(self, arch, mod, input_size, m, d_model, dropout, multclass = False, NS_model=0):
        super(ae_backbone, self).__init__()
        self.arch = arch
        self.mod = mod
        self.multclass = multclass
        self.m = m
        self.relu = nn.ReLU()

        self.fe1 = FE(mod, NS_model, input_size, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)


        if mod == "trx":
            self.out1 = nn.Linear(d_model, d_model)
            self.out2 = nn.Linear(d_model, 1)
        else:
            if multclass:
                self.out = nn.Linear(d_model, 2**m)
            else:
                self.out = nn.Linear(d_model, 2*m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        enc_out = self.fe1(src)
        enc_out = self.norm1(enc_out)
        if self.mod == "rec":
            enc_out = self.out(enc_out)
        else:
            enc_out = self.out1(enc_out)
            enc_out = self.out2(enc_out)
   
        if self.mod == "rec":
            if self.multclass == False:
                batch = enc_out.size(0)
                ell = enc_out.size(1)
                enc_out = enc_out.contiguous().view(batch, ell*self.m,2)
                output = F.softmax(enc_out, dim=-1)
            else:
                output = F.softmax(enc_out, dim=-1)
        else:
            # encoders
            output = enc_out
        return output
#######################################################################################




################################### security layer ###################################
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
    """ 
    Generates exponent and log tables for GF(2^q) using the given primitive polynomial. 
    """
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
    
    s_tensor = torch.tensor([int(''.join(map(str, s)), 2)], dtype=torch.int64, device=device)
    s_inverse = gf2_inv(s_tensor, exp_table, log_table, q, device) 
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

########################################################################################


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args

        self.Tmodel = ae_backbone(args.arch, "trx",  args.q + 3*(self.args.T-1), args.q, args.d_model_trx, args.dropout, args.multclass, args.enc_NS_model)

        self.Rmodel1 = ae_backbone(args.arch, "rec", args.T, args.q, args.d_model_rec, args.dropout, args.multclass, args.dec_NS_model)
        
        ########## Power Reallocation  ###############
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def power_constraint(self, inputs, isTraining, train_mean, train_std, idx = 0): # Normalize through batch dimension

        if isTraining == 1 or train_mean is None:
            # training
            this_mean = torch.mean(inputs, 0)   
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # use stats from training
            this_mean = train_mean[idx]
            this_std = train_std[idx]

        outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
  
        return outputs, this_mean.detach(), this_std.detach()

    def forward(self, train_mean, train_std, secreVec_1, fwd_noise_par1, fb_noise_par1, fwd_noise_par2, fb_noise_par2, isTraining = 1):
        """
        1: Bob
        2: Eve
        """
        num_samples = secreVec_1.shape[0]
        combined_noise_par1 = fwd_noise_par1 + fb_noise_par1 
        combined_noise_par2 = fwd_noise_par2 + fb_noise_par2
        for idx in range(self.args.T): 
            if idx == 0:
                src = torch.cat([secreVec_1, torch.zeros(num_samples, args.ell, self.args.T-1).to(self.args.device), torch.zeros(num_samples, args.ell, self.args.T-1).to(self.args.device), torch.zeros(num_samples, args.ell, self.args.T-1).to(self.args.device)], dim=2)
            elif idx == self.args.T-1:
                src = torch.cat([secreVec_1, parity_all, parity_fb1, parity_fb2],dim=2)
            else:
                src = torch.cat([secreVec_1, parity_all, torch.zeros(num_samples, args.ell, self.args.T-(idx+1) ).to(self.args.device), parity_fb1, torch.zeros(num_samples, args.ell, self.args.T-(idx+1) ).to(self.args.device), parity_fb2, torch.zeros(num_samples, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
            output = self.Tmodel(src)
                        
            ############# Generate the output ###################################################
            
            parity, x_mean, x_std = self.power_constraint(output, isTraining, train_mean, train_std, idx)

            if self.args.reloc == 1:
                parity = self.total_power_reloc(parity,idx)

            if idx == 0:
                parity_fb1 = parity + combined_noise_par1[:,:,idx].unsqueeze(-1)
                parity_fb2 = parity + combined_noise_par2[:,:,idx].unsqueeze(-1)
                parity_all = parity
                received1 = parity + fwd_noise_par1[:,:,idx].unsqueeze(-1)
                received2 = parity + fwd_noise_par2[:,:,idx].unsqueeze(-1)
                x_mean_total, x_std_total = x_mean, x_std
            else:
                parity_fb1 = torch.cat([parity_fb1, parity + combined_noise_par1[:,:,idx].unsqueeze(-1)],dim=2) 
                parity_fb2 = torch.cat([parity_fb2, parity + combined_noise_par2[:,:,idx].unsqueeze(-1)],dim=2) 
                parity_all = torch.cat([parity_all, parity], dim=2)     
                received1 = torch.cat([received1, parity + fwd_noise_par1[:,:,idx].unsqueeze(-1)], dim = 2)
                received2 = torch.cat([received2, parity + fwd_noise_par2[:,:,idx].unsqueeze(-1)], dim = 2)

                x_mean_total = torch.cat([x_mean_total, x_mean], dim = 0)
                x_std_total = torch.cat([x_std_total, x_std], dim = 0)

        decSeq1 = self.Rmodel1(received1) 
        
        return decSeq1, received1, received2, x_mean_total, x_std_total, parity_all


def train_model(model, args, logging, initial_weights_path = None):
    if initial_weights_path and os.path.exists(initial_weights_path):
        print(f"Loading initial weights from {initial_weights_path}")
        checkpoint = torch.load(initial_weights_path, map_location=args.device)
        model.load_state_dict(checkpoint)
    else:
        print("No initial weights provided, training from scratch")

    # security layer
    exp_table, log_table = get_table(args.q)

    ######################choose the seed################################################
    s = [1, 0, 1]
    # [1, 1, 0, 1]
    batch_size_mine = 5000
    
    mine_kwargs = {
        'loss': 'mine',
        'alpha': 0.01,
        'lr': 1e-3,
        'gamma':0.98,
        'batch_size': batch_size_mine,
        'train_loader': None,
        'test_loader': None
    }

    mi_estimator = MutualInformationEstimator(args.K, args.T, **mine_kwargs).to(args.device)
    mine_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-3)

    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    logging.info("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []
    flag = 0
    
    pbar = tqdm(range(args.totalbatch))
    train_mean = torch.zeros(args.T, 1).to(args.device)
    train_std = torch.zeros(args.T, 1).to(args.device)
    
    pktErrors1 = 0
    secrey_pktErrors1 = 0
    min_mi = 1000000

    for eachbatch in pbar:
        mVec_1 = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
        bVec_1 = torch.randint(0, 2, (args.batchSize, args.ell, args.q - args.m))

        combedVec_1 = torch.cat([mVec_1, bVec_1], dim = 2)

        s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, s, exp_table, log_table, args.device)
        
        snr2_bob = args.snr2_bob
        snr2_eve = args.snr2_eve
        if eachbatch < 0:
            snr1_bob=4* (1-eachbatch/(args.core * 30000))+ (eachbatch/(args.core * 30000)) * args.snr1_bob
            snr1_eve=4* (1-eachbatch/(args.core * 30000))+ (eachbatch/(args.core * 30000)) * args.snr1_eve
        else:
            snr1_bob=args.snr1_bob
            snr1_eve=args.snr1_eve
        
        std1_bob = 10 ** (-snr1_bob * 1.0 / 10 / 2) #forward snr
        std1_eve = 10 ** (-snr1_eve * 1.0 / 10 / 2) #forward snr

        std2_bob = 10 ** (-snr2_bob * 1.0 / 10 / 2) #feedback snr
        std2_eve = 10 ** (-snr2_eve * 1.0 / 10 / 2) #feedback snr

        # Noise values for the parity bits
        fwd_noise_bob = torch.normal(0, std=std1_bob, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_bob = torch.normal(0, std=std2_bob, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fwd_noise_eve = torch.normal(0, std=std1_eve, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_eve = torch.normal(0, std=std2_eve, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2_bob >= 100:
            fb_noise_bob = 0* fb_noise_bob
        if args.snr2_eve >= 100:
            fb_noise_eve = 0* fb_noise_eve

        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        preds1, received_bob, received_eve, batch_mean, batch_std, parity_all = model(None, None, secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), isTraining=1)
        if batch_mean is not None:
            train_mean += batch_mean
            train_std += batch_std 
   
        # Phase 1: Train MINE for 50 epochs 
        m_flat = mVec_1.view(args.batchSize, -1)
        z_flat = received_eve.view(args.batchSize, -1)
        if eachbatch % 50 == 0:  
            mi_estimator.train()
            dataset = TensorDataset(m_flat.detach(), z_flat.detach())
            train_loader = DataLoader(dataset, batch_size=batch_size_mine, shuffle=True)
            
            for _ in range(30): 
                for x, z in train_loader:
                    mine_optimizer.zero_grad()
                    loss = mi_estimator(x, z)
                    loss.backward()
                    mine_optimizer.step()



        # Phase 2: Update encoder-decoder with trade-off loss
        args.optimizer.zero_grad()
        mi_estimator.eval()
        ys1 = secreVec_1_decimal.long().contiguous().view(-1)
        preds1 = preds1.contiguous().view(-1, preds1.size(-1)) 
        preds1 = torch.log(preds1)
        ce_loss = F.nll_loss(preds1, ys1.to(args.device))
        
      
        mi_estimate = -mi_estimator(m_flat, z_flat)
        mi_estimate_bits = mi_estimate / math.log(2)
        print("MI estimate nats:", mi_estimate.item())
        print("MI estimate bits:", mi_estimate_bits.item())
       
        total_loss = ce_loss + args.beta * mi_estimate_bits

       
        total_loss.backward()

        probs1, decodeds1 = preds1.max(dim=1)
        decisions1 = decodeds1 != ys1.to(args.device)

        pktErrors1 += decisions1.view(args.batchSize, args.ell).sum(1).count_nonzero()
        PER1 = pktErrors1 / (eachbatch + 1) / args.batchSize

        secre_decode_input = decodeds1.reshape(-1, 1, 1)
        secre_decode_output = secrecy_decode(args.q, secre_decode_input, s_tensor, args.m, exp_table, log_table)
        secre_decode_output = secre_decode_output.to(device)
        mVec_1 = mVec_1.to(device)
        secrey_pktErrors1 += torch.sum((mVec_1 != secre_decode_output).any(dim=2).float())

        # if mi_estimate.item() < min_mi:
        #     min_mi = mi_estimate.item()
        #     print("min mi:", min_mi)
        # elif mi_estimate.item() > min_mi + 0.7:
        #     break
            
            
        ####################### Gradient Clipping optional ###########################
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ##############################################################################
        args.optimizer.step()
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals) # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy ##############################
        if eachbatch%100 == 0:
            with torch.no_grad():
                print(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve}, BS {args.batchSize}, ce_loss {round(ce_loss.item(), 8)}, PER1 {round(PER1.item(), 10)}, MI(M;Z): {mi_estimate.item():.4f}")
                logging.info(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve}, BS {args.batchSize}, ce_loss {round(ce_loss.item(), 8)}, PER1 {round(PER1.item(), 10)}, MI(M;Z): {mi_estimate.item():.4f}")		
   
                # test with large batch every 1000 batches
                print("Testing started: ... ")
                logging.info("Testing started: ... ")
                # change batch size to 10x for testing
                args.batchSize = int(args.batchSize*10)
                EvaluateNets(model, None, None, args, logging)
                args.batchSize = int(args.batchSize/10)
                print("... finished testing")
    
        ####################################################################################
        if np.mod(eachbatch, args.core * 100) == args.core - 1:
            epoch_loss_record.append(loss.item())
            if not os.path.exists(weights_folder):
                os.mkdir(weights_folder)
            torch.save(epoch_loss_record, f'{weights_folder}/loss')

        if np.mod(eachbatch, args.core * 100) == args.core - 1:# and eachbatch >= 80000:
            if not os.path.exists(weights_folder):
                os.mkdir(weights_folder)
            saveDir = f'{weights_folder}/model_weights' + str(eachbatch) + '.pt'
            torch.save(model.state_dict(), saveDir)
        pbar.update(1)
        pbar.set_description(f"GBAF train stats: batch#{eachbatch}, ce_loss {round(ce_loss.item(), 8)}")

        # kill the training if the loss is nan
        if np.isnan(ce_loss.item()):
            print("Loss is nan, killing the training")
            logging.info("Loss is nan, killing the training")
            break
  
    pbar.close()

    if train_mean is not None:
        train_mean = train_mean / args.totalbatch
        train_std = train_std / args.totalbatch	# not the best way but numerically stable
      
    return train_mean, train_std

def EvaluateNets(model, train_mean, train_std, args, logging):
    exp_table, log_table = get_table(args.q)
    s = [1, 0, 1]
    # [1, 1, 0, 1] 
   
    if args.train == 0:
        path = f'{weights_folder}/model_weights{args.totalbatch-10}.pt'
        print(f"Using model from {path}")
        logging.info(f"Using model from {path}")
    
        checkpoint = torch.load(path,map_location=args.device)
    
        # ======================================================= load weights
        model.load_state_dict(checkpoint)
        model = model.to(args.device)
    model.eval()
    map_vec = 2**(torch.arange(args.m))

    args.numTestbatch = 100000000
    

    symErrors1 = 0
    pktErrors1 = 0

    secrey_pktErrors1 = 0

 
    start_time = time.time()

    for eachbatch in range(args.numTestbatch):

        mVec_1 = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
        bVec_1 = torch.randint(0, 2, (args.batchSize, args.ell, args.q - args.m))

        combedVec_1 = torch.cat([mVec_1, bVec_1], dim = 2)

        s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, s, exp_table, log_table, args.device)
      
        # generate n sequence 
        std1_bob = 10 ** (-args.snr1_bob * 1.0 / 10 / 2) #forward snr
        std1_eve = 10 ** (-args.snr1_eve * 1.0 / 10 / 2) #forward snr

        std2_bob = 10 ** (-args.snr2_bob * 1.0 / 10 / 2) #feedback snr
        std2_eve = 10 ** (-args.snr2_eve * 1.0 / 10 / 2) #feedback snr

        # Noise values for the parity bits
        fwd_noise_bob = torch.normal(0, std=std1_bob, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_bob = torch.normal(0, std=std2_bob, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fwd_noise_eve = torch.normal(0, std=std1_eve, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_eve = torch.normal(0, std=std2_eve, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2_bob >= 100:
            fb_noise_bob = 0* fb_noise_bob
        if args.snr2_eve >= 100:
            fb_noise_eve = 0* fb_noise_eve

        # feed into model to get predictions
        with torch.no_grad():
            preds1, received_bob, received_eve, _, _, parity_all = model(None, None, secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), isTraining=1)

            ## update error calculation      
            ys1 = secreVec_1_decimal.long().contiguous().view(-1)
            preds1 = preds1.contiguous().view(-1, preds1.size(-1)) 
            
            probs1, decodeds1 = preds1.max(dim=1)
            decisions1 = decodeds1 != ys1.to(args.device)
        

            symErrors1 += decisions1.sum()
            SER1 = symErrors1 / (eachbatch + 1) / args.batchSize / args.ell
            

            pktErrors1 += decisions1.view(args.batchSize, args.ell).sum(1).count_nonzero()
            PER1 = pktErrors1 / (eachbatch + 1) / args.batchSize
           
            
            num_batches_ran = eachbatch + 1
            num_pkts = num_batches_ran * args.batchSize	

            secre_decode_input = decodeds1.reshape(-1, 1, 1)
            ########################################################################################################################
            secre_decode_output = secrecy_decode(args.q, secre_decode_input, s_tensor, args.m, exp_table, log_table)            ######################################################################################################################## 
            ########################################################################################################################
            secre_decode_output = secre_decode_output.to(device)
            mVec_1 = mVec_1.to(device)
            secrey_pktErrors1 += torch.sum((mVec_1 != secre_decode_output).any(dim=2).float())
            PER1_secre = secrey_pktErrors1/ (eachbatch + 1) / args.batchSize
            
            
            if eachbatch%100 == 0:
                print(f"\nwiretap test stats: batch#{eachbatch}, SER1 {round(SER1.item(), 10)}, numErr1 {symErrors1.item()}, num_pkts1 {num_pkts:.2e}, PER1_secre {PER1_secre:.2e}")
                logging.info(f"\nwiretap test stats: batch#{eachbatch}, SER1 {round(SER1.item(), 10)}, numErr {symErrors1.item()}, num_pkts {num_pkts:.2e}, PER1_secre {PER1_secre:.2e}")
                print(f"Time elapsed: {(time.time() - start_time)/60} mins")
                logging.info(f"Time elapsed: {(time.time() - start_time)/60} mins")
            if args.train == 1:
                min_err = 20
            else:
                min_err = 100
            if symErrors1 > min_err or (args.train == 1 and num_batches_ran * args.batchSize * args.ell > 1e8):
                print(f"\nwiretap test stats: batch#{eachbatch}, SER {round(SER1.item(), 10)}, numErr {symErrors1.item()}")
                logging.info(f"\nwiretap test stats: batch#{eachbatch}, SER {round(SER1.item(), 10)}, numErr {symErrors1.item()}")
                break

            

    SER1 = symErrors1.cpu() / (num_batches_ran * args.batchSize * args.ell)
    PER1 = pktErrors1.cpu() / (num_batches_ran * args.batchSize)
    secrey_PER1 = secrey_pktErrors1.cpu()/ (num_batches_ran * args.batchSize)
    print(f"Final test SER1 = {torch.mean(SER1).item()}, secrecy SER1 = {secrey_PER1}, at snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve} for rate {args.q}/{args.T}")
    print(f"Final test PER1 = {torch.mean(PER1).item()}, secrecy SER1 = {secrey_PER1}, at snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve} for rate {args.q}/{args.T}")
    logging.info(f"Final test SER1 = {torch.mean(SER1).item()}, at snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve} for rate {args.m}/{args.T}")
    logging.info(f"Final test PER1 = {torch.mean(PER1).item()}, at snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve} for rate {args.m}/{args.T}")

if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser(True)
    ########### path for saving model checkpoints ################################
    args.saveDir = f'weights/model_weights{args.totalbatch-101}.pt'  # path to be saved to
    ################## Model size part ###########################################
    args.d_model_trx = args.d_k_trx # total number of features
    args.d_model_rec = args.d_k_rec # total number of features
 
    # fix the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
   
 
    model = AE(args).to(args.device)
    # check if device contains the string 'cuda'
    if 'cuda' in args.device:
        # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        torch.backends.cudnn.benchmark = True
 
    # ======================================================= Initialize the model
    model = AE(args).to(args.device)
  
    # configure the logging
    folder_str = f"T_{args.T}/pow_{args.reloc}/{args.batchSize}/{args.lr}/"
    sim_str = f"K_{args.K}_m_{args.m}_q{args.q}_snr1bob_{args.snr1_bob}_snr1eve_{args.snr1_eve}_beta_{args.beta}"
 
    parent_folder = f"wiretap_results/N_{args.enc_NS_model}_{args.dec_NS_model}_d_{args.d_k_trx}_{args.d_k_rec}/snr2bob_{args.snr2_bob}_snr2eve_{args.snr2_eve}_beta{args.beta}/seed_{args.seed}"
    log_file = f"log_{sim_str}.txt"
    log_folder = f"{parent_folder}/logs/gbaf_{args.arch}_{args.features}/{folder_str}"
    log_file_name = os.path.join(log_folder, log_file)
 
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)

    global weights_folder
    weights_folder = f"{parent_folder}/weights/gbaf_{args.arch}_{args.features}/{folder_str}/{sim_str}/"
    os.makedirs(weights_folder, exist_ok=True)


    # ======================================================= run
    if args.train == 1:
        if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
            args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
            lambda1 = lambda epoch: (1-epoch/args.totalbatch)
            args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)

        # print the model summary
        print(model)
        logging.info(model)
  
        # print the number of parameters in the model that need to be trained
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {num_params}")
        logging.info(f"Total number of trainable parameters: {num_params}")
  
        # print num params in Tmodel
        num_params = sum(p.numel() for p in model.Tmodel.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters in Tmodel: {num_params}")
        logging.info(f"Total number of trainable parameters in Tmodel: {num_params}")
        # print num params in Rmodel
        num_params = sum(p.numel() for p in model.Rmodel1.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters in Rmodel1: {num_params}")
        logging.info(f"Total number of trainable parameters in Rmodel1: {num_params}")

        initial_weights_path = "wiretap_results/N_3_3_d_16_16/snr2bob_100_snr2eve_100/seed_144/weights/gbaf_1xfe_fy/T_9/pow_1/100000/0.001//K_3_m_3_q3_snr1bob_-1_snr1eve_-1//model_weights200000.pt"
        # "wiretap_results/N_3_3_d_16_16/snr2bob_100_snr2eve_100/seed_144/weights/gbaf_1xfe_fy/T_9/pow_1/100000/0.001//K_3_m_3_q4_snr1bob_-1_snr1eve_-1//model_weights200000.pt"

        train_mean, train_std = train_model(model, args, logging, initial_weights_path)

        # stop training and test
        args.train = 0
        args.batchSize = int(args.batchSize*10)
        start_time = time.time()
  
        print("\nInference after training: ... ")
        logging.info("\nInference after training: ... ")
        EvaluateNets(model, None, None, args, logging)
        args.batchSize = int(args.batchSize/10)
  
        end_time = time.time()
        tot_time_mins = (end_time - start_time) / 60
        print(f"\nTime for testing: {tot_time_mins}")
        logging.info(f"\nTime for testing: {tot_time_mins}")

    ## Inference
    print("\nInference using trained model and stats from large dataset: ... ")
    logging.info("\nInference using trained model and stats from large dataset: ... ")

    path = f'{weights_folder}/model_weights{args.totalbatch-10}.pt'
    
    print(f"\nUsing model from {path}")
    logging.info(f"\nUsing model from {path}")
 
    # use one very large batch to compute mean and std
    large_bs = int(1e6)
    args.batchSize = large_bs
    checkpoint = torch.load(path,map_location=args.device)

    # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()

    
    mVec_1 = torch.randint(0, 2, (large_bs, args.ell, args.m))
    bVec_1 = torch.randint(0, 2, (large_bs, args.ell, args.q - args.m))
    combedVec_1 = torch.cat([mVec_1, bVec_1], dim = 2)

    exp_table, log_table = get_table(args.q)
    s = [1, 0, 1]
    # [1, 1, 0, 1]
  
    s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, s, exp_table, log_table, args.device)

     
    # generate n sequence 
    std1_bob = 10 ** (-args.snr1_bob * 1.0 / 10 / 2) #forward snr
    std1_eve = 10 ** (-args.snr1_eve * 1.0 / 10 / 2) #forward snr

    std2_bob = 10 ** (-args.snr2_bob * 1.0 / 10 / 2) #feedback snr
    std2_eve = 10 ** (-args.snr2_eve * 1.0 / 10 / 2) #feedback snr

    # Noise values for the parity bits
    fwd_noise_bob = torch.normal(0, std=std1_bob, size=(args.batchSize, args.ell, args.T), requires_grad=False)
    fb_noise_bob = torch.normal(0, std=std2_bob, size=(args.batchSize, args.ell, args.T), requires_grad=False)
    fwd_noise_eve = torch.normal(0, std=std1_eve, size=(args.batchSize, args.ell, args.T), requires_grad=False)
    fb_noise_eve = torch.normal(0, std=std2_eve, size=(args.batchSize, args.ell, args.T), requires_grad=False)
    if args.snr2_bob >= 100:
        fb_noise_bob = 0* fb_noise_bob
    if args.snr2_eve >= 100:
        fb_noise_eve = 0* fb_noise_eve

    # feed into model to get predictions
    with torch.no_grad():
        preds1, received_bob, received_eve, train_mean, train_std, parity_all = model(None, None, secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), isTraining=1)

    EvaluateNets(model, train_mean, train_std, args, logging)
    estimate_mi(s, model, train_mean, train_std, args, logging)


    