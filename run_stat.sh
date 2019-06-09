
OMP_NUM_THREADS=32
mkdir -p log

for n in `seq 100 100 3000`
do 
m=printf "%04d" ${n}

python script/generate_problem.py -n ${n} -c 4.26 -p 1
log=log/log${m}.txt 
result=log/result${m}.txt 
echo -n "" >${log}

for i in `seq 1 10`
do
time -p ./test data/problem000.dat ${i} 2>> ${log}
done

grep real ${log} >${result}

done
#time -p Sparrow2Riss-2018/bin/SparrowToRiss.sh ./data/problem000.dat 123 ./tmp 

