
export OMP_NUM_THREADS=32
mkdir -p log_abd
mkdir -p log_riss

for n in `seq 10000 10000 100000`
do 
m=`printf "%06d" ${n}`

python script/generate_problem.py -n ${n} -c 4.26 -p 1
log1=log_abd/log${m}.txt 
result1=log_abd/result${m}.txt 

log2=log_riss/log${m}.txt 
result2=log_riss/result${m}.txt 

echo -n "" >${log1}
echo -n "" >${log2}

for i in `seq 1 10`
do
echo ${log1}
echo ${log2}
time -p ./test ./data/problem000.dat ${i} 1000 10 2>> ${log1}
time -p Sparrow2Riss-2018/bin/SparrowToRiss.sh ./data/problem000.dat ${i} ./tmp 2>>${log2}
done

grep real ${log1} >${result1}
grep real ${log2} >${result2}

done

#time -p python python/tensorized_sat_solver_sp.py ./data/problem000.dat  --sample_size 100 --max_itr 300

