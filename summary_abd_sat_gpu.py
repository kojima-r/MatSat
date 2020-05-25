import numpy as np
import time
from scipy.sparse import *
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, default=None, nargs="+", help="filenames")
parser.add_argument("--seed", type=int, default=123, help="random seed")
parser.add_argument("--num_clauses", "-m", type=int, default=None, help="|clause|")
args = parser.parse_args()

np.random.seed(123)
data={}
for filename in args.input:
    x = []
    i=0
    final_error=""
    time_real=""
    for a in open(filename):
        m=re.match("real",a)
        if m:
            time_real=a.strip().split(" ")[1]
        m=re.match("i=([0-9]+)\t",a)
        if m:
            x=m.group(1)
            j=int(x)
            if j > i:
                i=j
        m=re.match("final error = ([0-9\.]+)",a)
        if m:
            x=m.group(1)
            final_error=x
    name,ext=os.path.splitext(filename)
    name,var=os.path.splitext(name)
    var=var.strip(".")
    if var!="":
        print("\t".join([filename,name,var, str(i), time_real, final_error]))
        if name not in data:
            data[name]=[]
        data[name].append([filename,name,var, str(i), time_real, final_error])
for name,vec in data.items():
    t=[float(el[4]) for el in vec]
    e=[]
    cnt=0
    for el in vec:
        if el[5]!="":
            e.append(int(el[5]))
            if int(el[5])==0:
                cnt+=1
        else:
            e.append(1000)
    min_e=min(e)
    min_index=e.index(min_e)
    min_t=t[min_index]
    print(name,min_e,min_t,cnt)
        
