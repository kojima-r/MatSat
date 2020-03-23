mkdir -p tmp/

#Sparrow2Riss-2018/bin/SparrowToRiss.sh 3SAT_inst500 123 ./tmp

#time -p Sparrow2Riss-2018/bin/SparrowToRiss.sh ./data/problem000.dat 123 ./tmp

for f in `ls data/*`
do

echo ${f}
time -p Sparrow2Riss-2018/bin/SparrowToRiss.sh ${f} 123 ./tmp

done

