import numpy as np
import time
import argparse
from scipy.sparse import *


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str,
		help='cnf file (3sat)')
parser.add_argument('--seed', type=int,
		default=123,
		help='random seed')
parser.add_argument('--sample_size', type=int,
		default=None,
		help='sample size')
parser.add_argument('--max_itr', type=int,
		default=None,
		help='max_itr')
args=parser.parse_args()
	
np.random.seed(args.seed)
fp=open(args.input)
data=[]
def convert_var_index(index):
	i=int(index)
	if i>0:
		return i-1
	else:
		return -i-1
for line in fp:
	if line[0]=="c":
		pass
	if line[0]=="p":
		arr=line.strip().split(" ")
		n=int(arr[2])
		m=int(arr[3])
	else:
		arr=line.strip().split(" ")
		if arr[3]=="0":
			data.append(list(map(int,arr[:3])))

data=np.array(data,dtype=np.int32)
data[data>0]=data[data>0]-1
data[data<0]=(data[data<0]*-1)+n-1

if args.sample_size is None:
	sample_size = input(" sample_size = 100 => ")
	if (len(sample_size)==0):
		sample_size = 100
	else:
		sample_size = int(sample_size )
else:
	sample_size = args.sample_size

if args.max_itr is None:
	max_itr = input(" max_itr = 100 => ")
	if (len(max_itr)==0):
		max_itr = 100
	else:
		max_itr =int(max_itr )
else:
	max_itr = args.max_itr

num_success = 0

l1 = 1.0
split = 50
alpha = 0.05
beta = 1.0
gamma = 1.0

count_itr=0

start = time.time()

Q = lil_matrix((m,2*n),dtype=np.float32)
#------------------------------------------------------------
for s,y in enumerate(data):
	Q[s,y] = 1
Q1 = Q[0:m,0:n]
Q2 = Q[0:m,n:2*n]

# dense to sparse
Q1=csr_matrix(Q1)
Q2=csr_matrix(Q2)
Q=csr_matrix(Q)

#------------------------------------------------------------

u = np.random.rand(n,1)
for i in range(sample_size):
	sym_diff = np.zeros(split)
	ls = np.linspace(min(u),max(u),split)
	for s in range(split):
		a = u>=ls[s]
		aa= np.concatenate((a,1-a),axis=0)
		b = (Q@aa)>0
		sym_diff[s] = sum(1-b)
	idx_sym = np.argmin(sym_diff)
	ini_error=sym_diff[idx_sym]
	mdl = (u>=ls[idx_sym])*1.0
	final_error = ini_error

	if (ini_error > 0):
		for j in range(max_itr):
			count_itr+=1
			uu= np.concatenate((u,1-u),axis=0)
			C = Q@uu
			#C:(90,1)
			#Q:(90,2*n)
			#E = 1-min(C,1)
			E=1.0-C
			E[E<0]=0.0
			F = u*(1-u)

			##---- choose J ---->
			# J(1):
			#
			#     J = (1/2)*( sumsq(E) + l1*sumsq(F) );
			#     Ja = (Q2-Q1)'*((C<=1).*E) + l1*F.*(1-2*u);
			#-------
			# J(2):
			#
			J1=np.sum(E)
			J2=l1*(np.sum(F**2))
			J = np.sum(E) + l1*(np.sum(F**2))
			CC=C<=1
			Ja = np.transpose((Q2-Q1))@CC + l1*F*(1-2*u)
			##-------

			##--- choose learning method --->
			# J+GD:
			#
			#     u_new = u - alpha*Ja;
			#-------
			# J+Newton's method:
			#
			y = np.sum(Ja**2)
			u_new = u
			if (y>0):
				u_new = u - gamma*(J/y)*Ja
			##-------

			u = beta*u_new + (1-beta)*u
			"""
			sym_diff = np.zeros(split)
			ls = np.linspace(min(u),max(u),split)
			for s in range(split):
				a = u>=ls[s]
				aa= np.concatenate((a,1-a),axis=0)
				b = (Q@aa)>0
				sym_diff[s] = np.sum(1-b)
			idx_sym = np.argmin(sym_diff)
			final_error=sym_diff[idx_sym]

			mdl = (u>=ls[idx_sym])*1.0
			#print("  k=%d  i=%d  j=%d: J=%0.4f(%0.4f,%0.4f,)  error=%d"%(k, i, j, J,J1,J2, final_error))
			"""
			# faster computation
			sym_diff = np.zeros((n,split))
			ls = np.tile(np.linspace(np.min(u),np.max(u),split),(n,1))
			a=u>=ls
			aa= np.concatenate((a,1-a),axis=0)
			b = (Q@aa)>0
			sym_diff = np.sum(1-b,axis=0)
			idx_sym = np.argmin(sym_diff)
			final_error=sym_diff[idx_sym]
			mdl = u>=ls[0,idx_sym]
			#
			if (final_error == 0):
				break
			if (J>1e6):
				break

		#endfor;
	#endif;
	print("  i=%d  error(%d -> %d)"%( i, ini_error, final_error)) 
	if (final_error==0):
		break
	u = 0.5*u+0.5*np.random.rand(n,1)
#endfor;

if (final_error==0):
	num_success+=1
#endfor;


#print(" success_rate=%0.2f\n"%(num_success/num_problem,))
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print ("itr_time:{0}".format(elapsed_time/count_itr) + "[sec]")

