#!/bin/bash

OUTDIR=${1}

for f in $OUTDIR/*.out;
do
  CSV=${f}_breakdown.csv
  echo $CSV
  echo "device,type,time" > $CSV
  grep "active" $f >> $CSV
  grep "idle" $f >> $CSV
  # sed -i -e 's/device-//g' $CSV
  sed -i -e 's/us//g' $CSV
  # sed -i -e 's/-compute-active/,compute/g' $CSV
  # sed -i -e 's/-data-active/,data/g' $CSV
  sed -i -e 's/\[/-/g' $CSV
  sed -i -e 's/\]//g' $CSV
  Rscript $PWD/breakdown.R $CSV ${f}_breakdown.pdf
  mv ${f}_breakdown.pdf $OUTDIR/pdfs/
done
