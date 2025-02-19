#!/bin/bash

log_path=$1
num_gpus=$2

grep "mapping" $log_path > "tmp.mapping" 
for ((i=0; i<num_gpus; ++i)); do
  grep "GPU\[$i\]" tmp.mapping > tmp.gpu$i.mapping.log
  rm gpu$i.mapping.log

  while IFS= read -r line; do
    msg_field=$(echo "$line" | awk -F'"message":' '{print $2}')
    msg_field=$(echo "$msg_field" | awk -F', "taskName":' '{print $1}')
    echo $msg_field >> gpu$i.mapping.log
  done < tmp.gpu$i.mapping.log
  rm tmp.gpu$i.mapping.log
done


grep "launching" $log_path > "tmp.launching" 
for ((i=0; i<num_gpus; ++i)); do
  grep "GPU\[$i\]" tmp.launching > tmp.gpu$i.launching.log
  rm gpu$i.launching.log

  while IFS= read -r line; do
    msg_field=$(echo "$line" | awk -F'"message":' '{print $2}')
    msg_field=$(echo "$msg_field" | awk -F', "taskName":' '{print $1}')
    echo $msg_field >> gpu$i.launching.log
  done < tmp.gpu$i.launching.log
  rm tmp.gpu$i.launching.log

done

rm tmp.mapping
rm tmp.launching
