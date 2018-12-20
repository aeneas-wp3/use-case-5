#!/bin/bash
#/bin/hostname
#/bin/date

mkdir prmon && tar xf prmon.tar.gz -C prmon --strip-components 1

./prmon/bin/prmon -p $$ -i 1 &

/bin/ls -la
echo $*
echo $PATH
#printenv
singularity --version
#export SINGULARITY_BINDPATH=$PWD:/mnt

singularity exec -H $PWD MLsing.simg mprof run --include-children --python $1 $2
/bin/ls -la
#/bin/ls -la /mnt/
mv *.dat ./run1.data
