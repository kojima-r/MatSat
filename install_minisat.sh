wget http://minisat.se/downloads/minisat-2.2.0.tar.gz
tar xvf ./minisat-2.2.0.tar.gz

cd minisat/
export MROOT=$(pwd) 
cd core
make

