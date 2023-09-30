resultdir=log_riss_2018Main_Chen
list=run_riss_2018Main_Chen.list

mkdir -p ${resultdir}
echo -n "" > ${list}
for f in `ls Data2018Main/Chen/*.cnf`
do
	b=`basename $f .cnf`
	log=${resultdir}/${b}.${count}.txt
	tmp=./tmp/${resultdir}/
	mkdir -p ${tmp}
	echo "time Sparrow2Riss-2018/bin/SparrowToRiss.sh ${f} 1234 ${tmp} > ${log} 2>&1" >> ${list}
done
echo "cat run_riss_2018Main_Chen.list | xargs -P<#process> -I{} -t bash -c '{}'"
