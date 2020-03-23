import numpy as np
import time
from scipy.sparse import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
		default=123,
		help='random seed')
parser.add_argument('--num_variables','-n', type=int,
		default=None,
		help='|variable|')
parser.add_argument('--num_clauses','-m', type=int,
		default=None,
		help='|clause|')

parser.add_argument('--clause_coef','-c', type=float,
		default=None,
		help='coef')

parser.add_argument('--num_problems','-p', type=int,
		default=None,
		help='|problem|')
args=parser.parse_args()
	
np.random.seed(123)
print("uniform random 3-sat problems");
if args.num_variables is not None:
	n =args.num_variables 
else:
	n = input(" |variable| = n = 20 => ")
	if (len(n)==0):
		n = 20
	else:
		n = int(n)

if args.num_clauses is not None:
	m =args.num_clauses 
elif args.clause_coef is not None:
	m =int(args.clause_coef*n)
else:
	m = input(" |clause| = m = 90 => ")
	if (len(m)==0):
		m = 90
	else:
		m=int(m)

if args.num_problems is not None:
	num_problem=args.num_problems 
else:
	num_problem = input(" num_problem = 100 =>  ")
	if (len(num_problem)==0):
		num_problem = 100
	else:
		num_problem =int(num_problem)
import os

os.makedirs('data/',exist_ok=True)

for k in range(num_problem):
	filename="data/problem%03d.%06d.dat"%(k,n)
	print("[SAVE]",filename)
	fp=open(filename,"w")
	mdl0 = np.random.rand(n,1)<0.5
	ans_mdl=np.concatenate([mdl0,1-mdl0],axis=0)
	clause_list=[]
	#------------------------------------------------------------
	for s in range(m):
		while(True):
			while(True):
				y = np.random.randint(0,2*n,size=3)
				z1 = abs(y[0]-y[1])
				z2 = abs(y[1]-y[2])
				z3 = abs(y[2]-y[0])
				if( z1>0 and z2>0 and z3>0 and z1!=n and z2!=n and z3!=n ):
					break
			#if( np.dot(q[s,:],ans_mdl[:,0])>0 ):
			if np.sum(ans_mdl[y,0])>0:
				clause_list.append(y)
				break
	fp.write("p cnf %d %d\n"%(n,m))
	for y in clause_list:
		y[y>=n]=-(y[y>=n]-n+1)
		y[y>=0]=y[y>=0]+1
		fp.write(" ".join(map(str,y)))
		fp.write(" 0\n")
