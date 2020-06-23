resultdir=log_gpu_2018Main_Chen

export CUDA_VISIBLE_DEVICES=3

mkdir -p ${resultdir}

for count in `seq 1 3`
do
for f in `ls Data2018Main/Chen/*.cnf`
do
	b=`basename $f .cnf`
	log=${resultdir}/${b}.${count}.txt
	echo "time -p ./abdsat_gpu2 ${f} ${count} 1000 100 512 0.5" > ${log}
	time -p ./abdsat_gpu2 ${f} ${count} 1000 100 512 0.5>> ${log} 2>&1
	sleep 1
done
done
 
