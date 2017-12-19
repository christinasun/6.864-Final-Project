import random
import torch
import numpy as np

def print_shape_variable(name, x):
    print "{} shape: {}".format(name,x.data.shape)
    return

def print_shape_tensor(name, x):
    print "{} shape: {}".format(name,x.shape)
    return

def set_seeds(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)