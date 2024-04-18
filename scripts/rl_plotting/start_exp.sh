#!/bin/bash

TARGET_DIR=$1

#APP_ARR=( "cholesky" "reduction" "stencil" "sweeps" )
#APP_ARR=( "reduction" )
APP_ARR=( "cholesky" )
#NUM_GPUS_ARR=( "4" "8" "16" "32" "64" "128" )
NUM_GPUS_ARR=( "32" )
#DATA_SIZE_ARR=( "0.5" "1" "2" )
DATA_SIZE_ARR=( "0.5" )
#BANDWIDTH_ARR=( "0.5" "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" )
#BANDWIDTH_ARR=( "0.5" "1" "4" "64" "256" )
BANDWIDTH_ARR=( "25" )
#MODE_ARR=(  "loadbalance" "eft_with_data" "random" "heft" )
MODE_ARR=(  "loadbalance" "eft_with_data" "random" "heft" )


#APP_ARR=( "cholesky" "reduction" "stencil" "sweeps" )
#APP_ARR=( "cholesky" )
#NUM_GPUS_ARR=( "4" "8" )
#DATA_SIZE_ARR=( "10" )
#BANDWIDTH_ARR=( "10" )
#MODE_ARR=( "heft" "loadbalance" "parla" "random" )

# File processing
if [ ! -f outputs ] ; then
  mkdir -p outputs/pdfs
fi

for APP in "${APP_ARR[@]}"; do
  script="test_"${APP}".py"
  for NUM_GPUS in "${NUM_GPUS_ARR[@]}"; do
    for DATA_SIZE in "${DATA_SIZE_ARR[@]}"; do
      for BANDWIDTH in "${BANDWIDTH_ARR[@]}"; do
        file_name=${APP}_${NUM_GPUS}_${DATA_SIZE}_${BANDWIDTH}
        noise_file_name=${file_name}".noise"
        order_file_name=${file_name}".order"
        csv_file_name=$file_name".csv"
        pdf_file_name=${file_name}".pdf"
        echo "Mode, Label, ExecutionTime," > $csv_file_name

        if [[ "$APP" == "cholesky" ]]; then
          python levelbylevel.py -g $NUM_GPUS -b 10 >> $csv_file_name
        elif [[ "$APP" == "reduction" ]]; then
          python homog_theory.py -g $NUM_GPUS -n 511 >> $csv_file_name
        elif [[ "$APP" == "stencil" ]]; then
          python homog_theory.py -g $NUM_GPUS -n 500 >> $csv_file_name
        elif [[ "$APP" == "sweeps" ]]; then
          python homog_theory.py -g $NUM_GPUS -n 400 >> $csv_file_name
        fi

        noise_generation=true
        SAVE_ORDER=true
        order_flag=" -so True"

        for i in 1; do
          echo "repeat:$i"
          for MODE in "${MODE_ARR[@]}"; do

            out_file_name=${file_name}_${MODE}_${i}".out"

            echo $out_file_name," mode:", $MODE
         
            echo $csv_file_name
            # Noise flag 
            if [ "$noise_generation" = true ]; then 
              noise_generation=false
              noise_flags=" -n True -sn True" 
            else
              noise_flags=" -ln True" 
            fi
            echo "python ${script} -e 1 -m ${MODE} ${noise_flags} -o random -g ${NUM_GPUS} -pb ${BANDWIDTH} -dd ${DATA_SIZE} ${order_flag}"
            python ${script} -e 1 -m ${MODE} ${noise_flags} -o random -g ${NUM_GPUS} -pb ${BANDWIDTH} -dd ${DATA_SIZE} ${order_flag} > $out_file_name
            grep "simtime" $out_file_name >> $csv_file_name

            mv $out_file_name outputs
            if [ "$SAVE_ORDER" == true ]; then
              SAVE_ORDER=false
              order_flag=" -lo True"
            fi
          done
        done

        echo "Rscript $PWD/simtime.R $csv_file_name $pdf_file_name"
        Rscript $PWD/simtime.R $csv_file_name $pdf_file_name
        mv $csv_file_name outputs
        mv $pdf_file_name outputs/pdfs
        mv replay.noise $noise_file_name
        mv replay.order $order_file_name
        mv $noise_file_name outputs
        mv $order_file_name outputs
      done
    done
  done
done
