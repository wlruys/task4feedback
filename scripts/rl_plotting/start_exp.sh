#!/bin/bash

TARGET_DIR=$1

APP_ARR=( "cholesky" "reduction" "stencil" "sweeps" )
NUM_GPUS_ARR=( "4" "8" "16" "32" "64" )
DATA_SIZE_ARR=( "0.5" "1" "2" )
BANDWIDTH_ARR=( "0.5" "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" )
MODE_ARR=( "heft" "loadbalance" "parla" "random" )


#APP_ARR=( "cholesky" "reduction" "stencil" "sweeps" )
APP_ARR=( "cholesky" )
NUM_GPUS_ARR=( "4" "8" )
DATA_SIZE_ARR=( "0.5" )
BANDWIDTH_ARR=( "512" )
MODE_ARR=( "heft" "loadbalance" "parla" "random" )


for APP in "${APP_ARR[@]}"; do
  script="test_"${APP}".py"
  for NUM_GPUS in "${NUM_GPUS_ARR[@]}"; do
    for DATA_SIZE in "${DATA_SIZE_ARR[@]}"; do
      for BANDWIDTH in "${BANDWIDTH_ARR[@]}"; do
        file_name=${APP}_${NUM_GPUS}_${DATA_SIZE}_${BANDWIDTH}
        csv_file_name=$file_name".csv"
        pdf_file_name=$file_name".pdf"
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
        for MODE in "${MODE_ARR[@]}"; do
        
          echo $csv_file_name
          # Noise flag 
          if [ "$noise_generation" = true ]; then 
            noise_generation=false
            noise_flags=" -n True -sn True" 
          else
            noise_flags=" -ln True" 
          fi
          echo "python ${script} -e 1 -m ${MODE} ${noise_flags} -o heft -g ${NUM_GPUS} -pb ${BANDWIDTH} -dd ${DATA_SIZE}"
          python ${script} -e 1 -m ${MODE} ${noise_flags} -o heft -g ${NUM_GPUS} -pb ${BANDWIDTH} -dd ${DATA_SIZE} > tmp.log
          grep "simtime" tmp.log >> $csv_file_name
        done

        echo "Rscript $PWD/simtime.R $csv_file_name $pdf_file_name"
        Rscript $PWD/simtime.R $csv_file_name $pdf_file_name

        # File processing
        if [ ! -f outputs ] ; then
          mkdir -p outputs/pdfs
        fi
        mv $csv_file_name outputs
        mv $pdf_file_name outputs/pdfs
      done
    done
  done
done
