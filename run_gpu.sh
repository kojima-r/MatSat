resultdir=log_gpu

mkdir -p ${resultdir}

#for n in `seq 160000 10000 300000`
for n in `seq 10000 10000 300000`
do 
    echo "n = "${n}
    m=`printf "%06d" ${n}`
    for i in `seq 0 1 9`
    do
      j=`printf "%03d" ${i}`
      log=${resultdir}/log${j}.${m}.txt
      time -p ./matsat_gpu ./data/problem${j}.${m}.dat 1 1000 100 > ${log} 2>&1
    done
done
 
