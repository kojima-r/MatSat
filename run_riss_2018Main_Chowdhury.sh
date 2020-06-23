resultdir=log_riss_2018Main_Chowdhury
list=run_riss_2018Main_Chowdhury.list

mkdir -p ${resultdir}
echo -n "" > ${list}
for f in `ls Data2018Main/Chowdhury/*.cnf`
do
	b=`basename $f .cnf`
	log=${resultdir}/${b}.${count}.txt
	tmp=./tmp/${resultdir}/
	mkdir -p ${tmp}
	echo "Sparrow2Riss-2018/bin/SparrowToRiss.sh ${f} 1234 ${tmp} > ${log} 2>&1" >> ${list}
done
echo "cat ${list} | xargs -P<#process> -I{} -t bash -c '{}'"
