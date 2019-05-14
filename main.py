import numpy as np
from lib.utils import loadData
from lib.conv_tasnet import ConvTasNet


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('data_dir')
parser.add_argument('--N', type=int, default=256)
parser.add_argument('--L', type=int, default=20)
parser.add_argument('--B', type=int, default=256)
parser.add_argument('--H', type=int, default=512)
parser.add_argument('--P', type=int, default=3)
parser.add_argument('--X', type=int, default=8)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--C', type=int, default=4)
parser.add_argument('--norm_type', type=str, default='gLN')
parser.add_argument('--causal', type=bool, default='False')
parser.add_argument('--mask_nonlinear', type=str, default='softmax')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()

if __name__ == '__main__':
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

	dataset = AudioDataset(args.data_dir, sr=44100, mode='train', verbose=1)
	print('DataLoading Done')

	model = ConvTasNet(**model_args)

	print(count_params(model))
	train(model, dataset, **train_args)