resultdir=log_matsat

mkdir -p ${resultdir}

#for n in `seq 160000 10000 300000`
#for n in `seq 100 300 1000`
#do 
n=100
    echo "n = "${n}
 
    m=`printf "%06d" ${n}`
    python script/generate_problem.py -n ${n} -c 4.26 -p 1000
#done
 
