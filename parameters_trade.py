import argparse

device = 'cuda:0'
K = 3
m = 3
ell = int(K/m)
memory = K
q = 4
T = 9


train = 1

seed = 154

arch = "1xfe" 
features = "fy" 
batchSize = int(100000)

def args_parser(jupyter_notebook):
    parser = argparse.ArgumentParser()

    # Sequence arguments
    parser.add_argument('--snr1_bob', type=int, default= 1, help="Transmission SNR for Bob")
    parser.add_argument('--snr1_eve', type=int, default= 1, help="Transmission SNR for Eve")
    parser.add_argument('--snr2_bob', type=int, default= 5, help="Feedback SNR for Bob")
    parser.add_argument('--snr2_eve', type=int, default= 5, help="Feedback SNR for Eve")
    parser.add_argument('--random_seed', type=list, default= [1, 1, 0, 1], help="random seed s")

    # secrecy constraint
    parser.add_argument('--tau_bits', type=float, default=0.9, help="secrecy_constraint bits")
    parser.add_argument('--beta_lr', type=float, default=0.01, help="beta learning rate")
    parser.add_argument('--mi_epochs', type=int, default=300, help="MI epochs")
    parser.add_argument('--mi_batch', type=int, default=5000, help="MI batch size")
    parser.add_argument('--cycle', type=int, default=50, help="The number of iterations when you re-train MI")


    parser.add_argument('--K', type=int, default=K, help="Sequence length")
    parser.add_argument('--m', type=int, default=m, help="Block size")
    parser.add_argument('--ell', type=int, default=ell, help="Number of bit blocks")
    parser.add_argument('--T', type=int, default=T, help="Number of interactions")
    parser.add_argument('--q', type=int, default=q, help="Length after secrecy")
    parser.add_argument('--seq_reloc', type=int, default=1)
    parser.add_argument('--memory', type=int, default=K)
    parser.add_argument('--core', type=int, default=1)
    parser.add_argument('--enc_NS_model', type=int, default=3)
    parser.add_argument('--dec_NS_model', type=int, default=3)
    parser.add_argument('--arch', type=str, default=arch)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--features', type=str, default=features)

    parser.add_argument('--d_k_trx', type=int, default=16, help="feature dimension")
    parser.add_argument('--d_k_rec', type=int, default=16, help="feature dimension")
    parser.add_argument('--dropout', type=float, default=0.0, help="prob of dropout")

    # Learning arguments
    parser.add_argument('--load_weights') # None
    parser.add_argument('--train', type=int, default= train)
    parser.add_argument('--reloc', type=int, default=1, help="w/ or w/o power rellocation")
    parser.add_argument('--totalbatch', type=int, default=8010, help="number of total batches to train")
    parser.add_argument('--batchSize', type=int, default=batchSize, help="batch size")
    parser.add_argument('--opt_method', type=str, default='adamW', help="Optimization method adamW,lamb,adam")
    parser.add_argument('--clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('--use_lr_schedule', type=bool, default = True, help="lr scheduling")
    parser.add_argument('--multclass', type=bool, default = True, help="bit-wise or class-wise training")
    parser.add_argument('--embedding', type=bool, default = False, help="vector embedding option")
    parser.add_argument('--embed_normalize', type=bool, default = True, help="normalize embedding")
    parser.add_argument('--belief_modulate', type=bool, default = True, help="modulate belief [-1,1]")
    parser.add_argument('--clas', type=int, default = 2, help="number of possible class for a block of bits")
    parser.add_argument('--rev_iter', type=int, default = 0, help="number of successive iteration")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('--device', type=str, default=device, help="GPU")
    args = parser.parse_args()

    if jupyter_notebook:
        args = parser.parse_args(args=[])   # for jupyter notebook
    else:
        args = parser.parse_args()    # in general

    return args
