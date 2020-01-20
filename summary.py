import numpy as np
import time
from scipy.sparse import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str,
		default=None,
		nargs="+",
		help='filenames')
parser.add_argument('--seed', type=int,
		default=123,
		help='random seed')
parser.add_argument('--num_clauses','-m', type=int,
		default=None,
		help='|clause|')
args=parser.parse_args()
	
np.random.seed(123)

for filename in args.input:
	x=[]
	for a in open(filename):
		arr=a.strip().split(" ")
		x.append(float(arr[1]))
	#line=[filename,str(np.mean(x)),str(np.std(x))]
	line=[filename,str(np.median(x)),str(np.std(x))]
	print("\t".join(line))

