import numpy as np
from conv_tasnet import ConvTasNet
from train import train
from data import AudioDataset
from utils import count_params
from predict import evaluate, separate

import torch

from argparse import ArgumentParser
from IPython import embed

parser = ArgumentParser()

parser.add_argument('--data_dir', type=str, default='../data/8000')
parser.add_argument('--sr',type=int, default=8000)
parser.add_argument('--N', type=int, default=256)
parser.add_argument('--L', type=int, default=20)
parser.add_argument('--B', type=int, default=256)
parser.add_argument('--H', type=int, default=512)
parser.add_argument('--P', type=int, default=3)
parser.add_argument('--X', type=int, default=8)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--C', type=int, default=4)
parser.add_argument('--voice_only', type=bool, default=False)
parser.add_argument('--norm_type', type=str, default='gLN')
parser.add_argument('--causal', type=bool, default='False')
parser.add_argument('--mask_nonlinear', type=str, default='softmax')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--seq_len', type=float, default=4, help='sequence length in seconds')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--evaluate', type=int, default=0)
parser.add_argument('--separate', type=int, default=0)
parser.add_argument('--model', type=str, default='./tasnet.pkl')
parser.add_argument('--cal_sdr', type=int, default=0, help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--output_dir', type=str, default='./output')

args = parser.parse_args()

if __name__ == '__main__':
    if args.voice_only: args.C = 2
    model_args = {
        'N': args.N,
        'L': args.L,
        'B': args.B,
        'H': args.H,
        'P': args.P,
        'X': args.X,
        'R': args.R,
        'C': args.C,
        'norm_type': args.norm_type,
        'causal': args.causal,
        'mask_nonlinear': args.mask_nonlinear
    }

    train_args = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }

    model = ConvTasNet(**model_args)

    if args.evaluate == 0 and args.separate == 0:
        dataset = AudioDataset(args.data_dir, sr=args.sr, mode='train', seq_len=args.seq_len, verbose=0, voice_only=args.voice_only)

        print('DataLoading Done')

        train(model, dataset, **train_args)
    elif args.evaluate == 1:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))

        dataset = AudioDataset(args.data_dir, sr=args.sr, mode='test', seq_len=args.seq_len, verbose=0, voice_only=args.voice_only)

        evaluate(model, dataset, args.batch_size, 0, args.cal_sdr)
    else:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))

        dataset = AudioDataset(args.data_dir, sr=args.sr, mode='test', seq_len=args.seq_len, verbose=0, voice_only=args.voice_only)

        separate(model, dataset, args.output_dir, sr=8000)
