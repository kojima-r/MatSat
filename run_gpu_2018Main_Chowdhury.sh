resultdir=log_gpu_2018Main_Chowdhury

export CUDA_VISIBLE_DEVICES=1

mkdir -p ${resultdir}

for count in `seq 1 10`
do
for f in `ls Data2018Main/Biere/*.cnf`
do
	b=`basename $f .cnf`
	log=${resultdir}/${b}.${count}.txt
	echo "time -p ./abdsat_gpu ${f} ${count} 10000 1000 128 0.7" > ${log}
	time -p ./abdsat_gpu ${f} ${count} 10000 1000 128 0.7 >> ${log} 2>&1
	sleep 1
done
done
 
