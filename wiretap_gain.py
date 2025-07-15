# WTC-Lightcode channel with CCE loss
import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from Feature_extractors_wire import FE
import copy
from parameters_gain import *
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

import math

def errors_bler(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    y_true = y_true.view(y_true.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)

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



################################### secrecy layer ###################################
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
        x <<= 1  
        if x & field_size:  
            x ^= prim_poly
    
    exp_table[field_size - 1] = 1  
    
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
        a_flat = a[mask].long()  
        b_flat = b[mask].long() 

        log_a = log_table[a_flat]
        log_b = log_table[b_flat]

        res[mask] = exp_table[(log_a + log_b) % (field_size - 1)]
    
    return res

def gf2_inv(a, exp_table, log_table, q, device):
    """
    Computes the multiplicative inverse of elements in GF(2^q).
    """
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
        
        ########## Power Reallocation ###############
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
        combined_noise_par1 = fwd_noise_par1 + fb_noise_par1 
        combined_noise_par2 = fwd_noise_par2 + fb_noise_par2
        for idx in range(self.args.T): 
            if idx == 0:
                src = torch.cat([secreVec_1, torch.zeros(self.args.batchSize, args.ell, self.args.T-1).to(self.args.device), torch.zeros(self.args.batchSize, args.ell, self.args.T-1).to(self.args.device), torch.zeros(self.args.batchSize, args.ell, self.args.T-1).to(self.args.device)], dim=2)
            elif idx == self.args.T-1:
                src = torch.cat([secreVec_1, parity_all, parity_fb1, parity_fb2],dim=2)
            else:
                src = torch.cat([secreVec_1, parity_all, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device), parity_fb1, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device), parity_fb2, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
            output = self.Tmodel(src)
                                    
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
    s = [1, 1, 0, 1] 
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

        # feed into model to get predictions
        preds1, received_bob, received_eve, batch_mean, batch_std, parity_all = model(None, None, secreVec_1_binary.to(args.device), fwd_noise_bob.to(args.device), fb_noise_bob.to(args.device), fwd_noise_eve.to(args.device), fb_noise_eve.to(args.device), isTraining=1)
        if batch_mean is not None:
            train_mean += batch_mean
            train_std += batch_std 

        args.optimizer.zero_grad()

        ys1 = secreVec_1_decimal.long().contiguous().view(-1)
        preds1 = preds1.contiguous().view(-1, preds1.size(-1)) 
        preds1 = torch.log(preds1)

        
        loss = F.nll_loss(preds1, ys1.to(args.device))

        probs1, decodeds1 = preds1.max(dim=1)
        decisions1 = decodeds1 != ys1.to(args.device)

        pktErrors1 += decisions1.view(args.batchSize, args.ell).sum(1).count_nonzero()
        PER1 = pktErrors1 / (eachbatch + 1) / args.batchSize


        loss.backward()


        secre_decode_input = decodeds1.reshape(-1, 1, 1)
        secre_decode_output = secrecy_decode(args.q, secre_decode_input, s_tensor, args.m, exp_table, log_table)
        secre_decode_output = secre_decode_output.to(device)
        mVec_1 = mVec_1.to(device)
        secrey_pktErrors1 += torch.sum((mVec_1 != secre_decode_output).any(dim=2).float())

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
        if eachbatch%100000 == 0:
            with torch.no_grad():
                print(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve}, BS {args.batchSize}, Loss {round(loss.item(), 8)}, PER1 {round(PER1.item(), 10)}")
                logging.info(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 bob {args.snr1_bob}, snr2 bob {args.snr2_bob}, snr1 eve {args.snr1_eve}, snr2 eve {args.snr2_eve}, BS {args.batchSize}, Loss {round(loss.item(), 8)}, PER1 {round(PER1.item(), 10)}")		
   
                # test with large batch every 1000 batches
                print("Testing started: ... ")
                logging.info("Testing started: ... ")
                # change batch size to 10x for testing
                args.batchSize = int(args.batchSize*10)
                EvaluateNets(model, None, None, args, logging)
                args.batchSize = int(args.batchSize/10)
                print("... finished testing")
    
        ####################################################################################
        if np.mod(eachbatch, args.core * 20000) == args.core - 1:
            epoch_loss_record.append(loss.item())
            if not os.path.exists(weights_folder):
                os.mkdir(weights_folder)
            torch.save(epoch_loss_record, f'{weights_folder}/loss')

        if np.mod(eachbatch, args.core * 20000) == args.core - 1:# and eachbatch >= 80000:
            if not os.path.exists(weights_folder):
                os.mkdir(weights_folder)
            saveDir = f'{weights_folder}/model_weights' + str(eachbatch) + '.pt'
            torch.save(model.state_dict(), saveDir)
        pbar.update(1)
        pbar.set_description(f"GBAF train stats: batch#{eachbatch}, Loss {round(loss.item(), 8)}")

        # kill the training if the loss is nan
        if np.isnan(loss.item()):
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
    ######################choose the seed################################################
    s =  [1, 1, 0, 1] 
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
    # check if device contains the string 'cuda'
    if 'cuda' in args.device:
        # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        torch.backends.cudnn.benchmark = True
 
    # ======================================================= Initialize the model
    model = AE(args).to(args.device)
  
    # configure the logging
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


        train_mean, train_std = train_model(model, args, logging, None)

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
    # path = 'wiretap_results/N_3_3_d_32_32/snr2bob_100_snr2eve_100/seed_666/weights/gbaf_1xfe_fy/T_9/pow_1/100000/0.001//K_3_m_3_snr1bob_0_snr1eve_0//model_weights80000.pt'
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
    ######################choose the seed################################################
    s = [1, 1, 0, 1] 
    print("seed", s)
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


    