# WTC-Lightcode channel with CCE loss, estimate mutual information
import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from Feature_extractors_wire import FE
import copy
from parameters_mine import *
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

from estimator.mi_classes import *
from estimator.mi_utils import *

from torch.utils.data import DataLoader, TensorDataset


def build_disc(dx, dy, divergence, architecture="deranged", mode="gauss", device="cuda"):
    if architecture == "deranged":
        simple_net = Net(input_dim=dx + dy, output_dim=1)
        return CombinedNet(simple_net, divergence).to(device)
    elif architecture == "joint":
        return ConcatCritic(dim=dx + dy, hidden_dim=256, layers=2, activation='relu', divergence=divergence).to(device)
    elif architecture == "separable":
        return SeparableCritic(dim=dx, hidden_dim=256, embed_dim=32, layers=2, activation='relu', divergence=divergence, mode=mode).to(device)
    else:
        raise ValueError("architecture must be 'deranged', 'joint', or 'separable'")


def forward_loss(disc, xb, yb, *, divergence, architecture, device="cuda", alpha=1.0, buffer=None):
    if architecture == "deranged":
        data_uv, data_u_v = data_generation_mi(xb, yb, device=device)   
        D1, D2 = disc(data_uv, data_u_v)                               
        if divergence == "MINE":
            loss, R, buffer = compute_loss_ratio(divergence, architecture, device, D_value_1=D1, D_value_2=D2, scores=None, buffer=buffer, alpha=alpha)
        else:
            loss, R = compute_loss_ratio(divergence, architecture, device,D_value_1=D1, D_value_2=D2, scores=None, buffer=None, alpha=alpha)
    else:
        scores = disc(xb, yb)
        if divergence == "MINE":
            loss, R, buffer = compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=scores, buffer=buffer, alpha=alpha)
        else:
            loss, R = compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=scores, buffer=None, alpha=alpha)

    I_hat = torch.mean(torch.log(R))
    return loss, I_hat, buffer


def train_mi(X, Y, divergence="KL", architecture="deranged", mode="gauss", epochs=300, batch_size=5000, lr=1e-3, alpha=1.0, device="cuda", tail_frac=0.10, do_val=True, val_batches=8):
    X = X.to(device).float(); Y = Y.to(device).float()
    dx, dy = X.shape[1], Y.shape[1]

    disc = build_disc(dx, dy, divergence, architecture, mode, device)
    opt  = optim.Adam(disc.parameters(), lr=lr)

    n = X.shape[0]; n_tr = int(0.8 * n)
    perm = torch.randperm(n, device=device)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]

    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True, drop_last=True)

    buffer = None 
    epoch_mi = []

    for _ in range(epochs):
        disc.train()
        batch_mis = []
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss, I_hat, buffer = forward_loss(disc, xb, yb, divergence=divergence, architecture=architecture, device=device, alpha=alpha, buffer=buffer)
            loss.backward()
            opt.step()
            batch_mis.append(float(I_hat.detach().cpu()))
        epoch_mi.append(float(np.mean(batch_mis)))


    I_val = None
    if do_val and len(Xte) >= batch_size:
        disc.eval()
        vals = []
        with torch.no_grad():
            for _ in range(val_batches):
                idx = torch.randint(0, Xte.shape[0], (batch_size,), device=device)
                xb, yb = Xte[idx], Yte[idx]
                _, I_hat_val, _ = forward_loss(disc, xb, yb, divergence=divergence, architecture=architecture, device=device, alpha=alpha, buffer=buffer)
                vals.append(float(I_hat_val.cpu()))
        I_val = float(np.mean(vals))
    
    return {
        "disc": disc,   
        "I_trace": epoch_mi,
        "I_val": I_val,
        "val_pair": (Xte, Yte)
    }


@torch.no_grad()
def eval_mi(disc, X, Y, divergence, architecture, device="cuda", alpha=1.0, batch_size=5000, eps=1e-8):
    X = X.to(device).float()
    Y = Y.to(device).float()
    dl = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False, drop_last=False)
    vals = []
    for xb, yb in dl:
        if architecture == "deranged":
            data_uv, data_u_v = data_generation_mi(xb, yb, device=device)
            D1, D2 = disc(data_uv, data_u_v)
            if divergence == "MINE":
                _, R, _ = compute_loss_ratio(divergence, architecture, device, D_value_1=D1, D_value_2=D2, scores=None, buffer=None, alpha=alpha)
            else:
                _, R = compute_loss_ratio(divergence, architecture, device, D_value_1=D1, D_value_2=D2, scores=None, buffer=None, alpha=alpha)
        else:
            scores = disc(xb, yb)
            if divergence == "MINE":
                _, R, _ = compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=scores, buffer=None, alpha=alpha)
            else:
                _, R = compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=scores, buffer=None, alpha=alpha)

        Ib = torch.mean(torch.log(R.clamp_min(eps)))
        vals.append(float(Ib.cpu()))
    return float(np.mean(vals)) if vals else float("nan")


def estimate_mi(model, train_mean, train_std, args, logging):
    exp_table, log_table = get_table(args.q)
    s = args.random_seed
    eve_list = [args.snr1_eve]

    divs   = ["MINE", "NWJ", 'HD', 'KL',  "GAN"] 
    arch   = "deranged"
    epochs = 300
    batch  = 5000
    alpha  = 1.0
    lr     = 1e-3
    tail_frac = 0.10  

    for eve_snr in eve_list:
        print(f"--------- eve snr {eve_snr} --------------------")
        if args.train == 0:
            path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'
            print(f"Using model from {path}")
            logging.info(f"Using model from {path}")
            checkpoint = torch.load(path, map_location=args.device)
            model.load_state_dict(checkpoint)
            model = model.to(args.device)
        model.eval()

        args.numTest = 100000
        mVec_1 = torch.randint(0, 2, (args.numTest, args.ell, args.m))
        bVec_1 = torch.randint(0, 2, (args.numTest, args.ell, args.q - args.m))
        combedVec_1 = torch.cat([mVec_1, bVec_1], dim=2)

        s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(
            args.q, combedVec_1, s, exp_table, log_table, args.device
        )

        std1_bob = 10 ** (-args.snr1_bob * 0.1 / 2)  # forward SNR (Bob)
        std1_eve = 10 ** (-eve_snr       * 0.1 / 2)  # forward SNR (Eve)
        std2_bob = 10 ** (-args.snr2_bob * 0.1 / 2)  # feedback SNR (Bob)
        std2_eve = 10 ** (-args.snr2_eve * 0.1 / 2)  # feedback SNR (Eve)

        # noise
        fwd_noise_bob = torch.normal(0, std=std1_bob, size=(args.numTest, args.ell, args.T), requires_grad=False)
        fb_noise_bob  = torch.normal(0, std=std2_bob,  size=(args.numTest, args.ell, args.T), requires_grad=False)
        fwd_noise_eve = torch.normal(0, std=std1_eve, size=(args.numTest, args.ell, args.T), requires_grad=False)
        fb_noise_eve  = torch.normal(0, std=std2_eve,  size=(args.numTest, args.ell, args.T), requires_grad=False)
        if args.snr2_bob >= 100: fb_noise_bob = 0 * fb_noise_bob
        if args.snr2_eve >= 100: fb_noise_eve = 0 * fb_noise_eve

        with torch.no_grad():
            preds1, received_bob, received_eve, _, _, parity_all = model(
                None, None,
                secreVec_1_binary.to(args.device),
                fwd_noise_bob.to(args.device),
                fb_noise_bob.to(args.device),
                fwd_noise_eve.to(args.device),
                fb_noise_eve.to(args.device),
                isTraining=1
            )

        message  = mVec_1.reshape(args.numTest, -1).to(args.device)
        bob      = received_bob.reshape(args.numTest, -1).to(args.device)
        eve      = received_eve.reshape(args.numTest, -1).to(args.device)

        device = args.device

        results_bob = {}
        print("------------- Mutual information: Bob -------------")
        for d in divs:
            print(f"----- method {d:>5s}")
            res = train_mi(message, bob, divergence=d, architecture=arch, epochs=epochs, batch_size=batch, lr=lr, alpha=alpha, device=device, tail_frac=tail_frac, do_val=True)
            results_bob[d] = res
            I_eval = eval_mi(res["disc"], message, bob, divergence=d, architecture=arch, device=device, batch_size=4096)
            print(f"Bob: {I_eval:.4f} nats, {I_eval/np.log(2):.4f} bits")

        results_eve = {}
        print("------------- Mutual information: eve -------------")
        for d in divs:
            print(f"----- method {d:>5s}")
            res = train_mi(message, eve, divergence=d, architecture=arch, epochs=epochs, batch_size=batch, lr=lr, alpha=alpha, device=device, tail_frac=tail_frac, do_val=True)
            results_eve[d] = res
            I_eval = eval_mi(res["disc"], message, eve, divergence=d, architecture=arch, device=device, batch_size=5000)
            print(f"Eve: {I_eval:.4f} nats, {I_eval/np.log(2):.4f} bits")



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

def PAMdedulation(noisy_theta, k):
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


################################### reliability encoding ###################################
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




################################### secrecy encoding ###################################
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
    # Primitive polynomial for GF(2^q) (assumes q=4, prim_poly=0b10011)
    # prim_poly = 0b10011 if q == 4 else NotImplemented  # 1 + x + x^4
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
    Secrecy encoding s^{-1} * combinedVec in GF(2^q).
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
    Secrecy decoding s * combinedVec_estimate in GF(2^q), taking k most significant bits.
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
        
        ########## Power Reallocation as in deepcode work ###############
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

def EvaluateNets(model, train_mean, train_std, args, logging):
    exp_table, log_table = get_table(args.q)
    if args.train == 0:
        path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'
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

        s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, args.random_seed, exp_table, log_table, args.device)
        
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
            secre_decode_output = secrecy_decode(args.q, secre_decode_input, s_tensor, args.m, exp_table, log_table)            ######################################################################################################################## 
            secre_decode_output = secre_decode_output.to(device)
            mVec_1 = mVec_1.to(device)
            secrey_pktErrors1 += torch.sum((mVec_1 != secre_decode_output).any(dim=2).float())
            PER1_secrecy = secrey_pktErrors1/ (eachbatch + 1) / args.batchSize
            
            if eachbatch%1000 == 0:
                print(f"\nwiretap test stats: batch#{eachbatch}, SER1 {round(SER1.item(), 10)}, numErr1 {symErrors1.item()}, num_pkts1 {num_pkts:.2e}, BLER {PER1_secrecy:.2e}")
                logging.info(f"\nwiretap test stats: batch#{eachbatch}, SER1 {round(SER1.item(), 10)}, numErr {symErrors1.item()}, num_pkts {num_pkts:.2e}, BLER {PER1_secrecy:.2e}")
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
    PER1_secrecy = secrey_pktErrors1.cpu()/ (num_batches_ran * args.batchSize)
    print(f"Final test SER1 = {torch.mean(SER1).item()}, BLER = {PER1_secrecy}, at snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve} for rate {args.q}/{args.T}")
    print(f"Final test PER1 = {torch.mean(PER1).item()}, BLER = {PER1_secrecy}, at snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve} for rate {args.q}/{args.T}")
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
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True
 
    model = AE(args).to(args.device)

    folder_str = f"T_{args.T}/pow_{args.reloc}/{args.batchSize}/{args.lr}/"
    sim_str = f"K_{args.K}_m_{args.m}_q{args.q}_snr1bob_{args.snr1_bob}_snr1eve_{args.snr1_eve}"
 
    parent_folder = f"wiretap_results/N_{args.enc_NS_model}_{args.dec_NS_model}_d_{args.d_k_trx}_{args.d_k_rec}/snr2bob_{args.snr2_bob}_snr2eve_{args.snr2_eve}/seed_{args.seed}"
 
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

        train_mean, train_std = train_model(model, args, logging)

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

    path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'

    print(f"\nUsing model from {path}")
    logging.info(f"\nUsing model from {path}")
 
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

    s_tensor, secreVec_1_decimal, secreVec_1_binary = secrecy_encode(args.q, combedVec_1, args.random_seed, exp_table, log_table, args.device)

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
        fb_noise_bob = 0 * fb_noise_bob
    if args.snr2_eve >= 100:
        fb_noise_eve = 0 * fb_noise_eve

    # feed into model to get predictions
    with torch.no_grad():
        preds1, received_bob, received_eve, train_mean, train_std, parity_all = model(None, None, secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), isTraining=1)

    EvaluateNets(model, train_mean, train_std, args, logging)
    estimate_mi(model, train_mean, train_std, args, logging)
