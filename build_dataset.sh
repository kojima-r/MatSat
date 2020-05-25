resultdir=log_abd

mkdir -p ${resultdir}

#for n in `seq 160000 10000 300000`
for n in `seq 10000 10000 300000`
do 
    echo "n = "${n}
 
    m=`printf "%06d" ${n}`
    python script/generate_problem.py -n ${n} -c 4.26 -p 10
done
 
