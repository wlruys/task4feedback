#!/bin/bash

TARGET_DIR=$1

##################### Configuration ###############################

# Applications
#APP_ARR=( "cholesky" "reduction" "stencil" "sweeps" )
#APP_ARR=( "reduction" )
APP_ARR=( "cholesky" "sweeps" )
#APP_ARR=( "cholesky" )
#APP_ARR=( "sweeps" )

# Number of GPUs
#NUM_GPUS_ARR=( "4" "8" "16" "32" "64" "128" )
NUM_GPUS_ARR=( "8" "32" )
#NUM_GPUS_ARR=( "4" )

# Data size (in GB)
#DATA_SIZE_ARR=( "0.5" "1" "2" )
DATA_SIZE_ARR=( "0.5" )

# Bandwidth (in GB/s)
#BANDWIDTH_ARR=( "0.5" "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" )
#BANDWIDTH_ARR=( "0.5" "1" "4" "64" "256" )
BANDWIDTH_ARR=( "25" "25000000" )

# Mapping policies
#MAPPING_POLICIES=(  "loadbalance" "eft_with_data" "random" "heft" )
MAPPING_POLICIES=( "loadbalance" "eft_without_data" "eft_with_data" "random" "heft" )
#MAPPING_POLICIES=( "eft_without_data" "eft_with_data" "heft" )

# Task creation order (All queues are FIFO)
# * heft: HEFT rank order
# * random: Random order
SORT_ARR=( "heft" "random" )


for APP in "${APP_ARR[@]}"; do
  # Script name to run
  script="test_"${APP}".py"
  for NUM_GPUS in "${NUM_GPUS_ARR[@]}"; do
    for DATA_SIZE in "${DATA_SIZE_ARR[@]}"; do
      for BANDWIDTH in "${BANDWIDTH_ARR[@]}"; do
        for SORT in "${SORT_ARR[@]}"; do

          # File processing
          if [ ! -f ${SORT}_outputs ] ; then
            mkdir -p ${SORT}_outputs/pdfs
          fi

          # Output file base name
          file_name=${APP}_${NUM_GPUS}_${DATA_SIZE}_${BANDWIDTH}
          # Created noise log 
          noise_file_name=${file_name}".noise"
          # Created task creation order log 
          order_file_name=${file_name}".order"
          # Total execution times for each configuration to be parsed by Rscripts
          csv_file_name=$file_name".csv"
          # Plot pdf name
          pdf_file_name=${file_name}".pdf"

          # Columns for output CSVs for plotting
          echo "Mode, Label, ExecutionTime," > $csv_file_name

          # Theory bounds (independent, serial)
          # NOTE that this uses # of tasks under the default settings
          if [[ "$APP" == "cholesky" ]]; then
            python levelbylevel.py -g $NUM_GPUS -b 10 >> $csv_file_name
					elif [[ "$APP" == "reduction" ]]; then
						python homog_theory.py -g $NUM_GPUS -n 511 >> $csv_file_name
					elif [[ "$APP" == "stencil" ]]; then
						python homog_theory.py -g $NUM_GPUS -n 500 >> $csv_file_name
					elif [[ "$APP" == "sweeps" ]]; then
						python homog_theory.py -g $NUM_GPUS -n 100 >> $csv_file_name
					fi

          # The first execution under a specific (app, # gpus, data size, bandwidth, sort)
          # will generate task noise and creation order.
          # Then, later executions under the same configuration with different policies reuse
          # the generated variabilities. 
          noise_generation=true
          SAVE_ORDER=true
          order_flag=" -so "

          # Repeat 3 times
          for i in 1 2 3; do
            for MAPPING in "${MAPPING_POLICIES[@]}"; do

              # All plain prints
              out_file_name=${file_name}_${MAPPING}_${i}".out"
              # Each mapping/launching logs:
              # * mapping: specify expected start~finish time for each task calculated
              #            at the mapping phase (NOTE that only EFT-based policies provide)
              # * launching: specify actual start~finish time for each task measured
              #            at the launching phase
              parsed_log_dir=${file_name}_${MAPPING}_${i}_logs
              mkdir $parsed_log_dir
              echo $out_file_name" mapping:" $MAPPING" iteration:" $i

              # Noise flag 
              if [ "$noise_generation" = true ]; then 
                noise_generation=false
                noise_flags=" -n -sn " 
              else
                noise_flags=" -ln " 
              fi

              # Execution
              echo "python ${script} -e 1 -m ${MAPPING} ${noise_flags} -o ${SORT} -g ${NUM_GPUS} -pb ${BANDWIDTH} -dd ${DATA_SIZE} ${order_flag} -d 'rr'"
              python ${script} -e 1 -m ${MAPPING} ${noise_flags} -o ${SORT} -g ${NUM_GPUS} -pb ${BANDWIDTH} -dd ${DATA_SIZE} ${order_flag} -d "rr" > $out_file_name
              # Redirect level-by-level time + simulated total execution times
              grep "simtime" $out_file_name >> $csv_file_name

              # Parse and split mapping/launching logs (check the above) for each device
              source log_parsing.sh logs/file.log ${NUM_GPUS}
              mv gpu*mapping.log ${parsed_log_dir}
              mv gpu*launching.log ${parsed_log_dir}

              # If those directories already exist, remove it since otherwise they are not moved
              # to the target directory but remains in the current directory
              rm -rf ${SORT}_outputs/$out_file_name
              rm -rf ${SORT}_outputs/$parsed_log_dir
              mv $out_file_name ${SORT}_outputs
              mv $parsed_log_dir ${SORT}_outputs

              # After this condition, all mapping policies use the same order
              if [ "$SAVE_ORDER" == true ]; then
                SAVE_ORDER=false
                order_flag=" -lo "
              fi
            done
          done

          echo "Rscript $PWD/simtime.R $csv_file_name $pdf_file_name"
          Rscript $PWD/simtime.R $csv_file_name $pdf_file_name
          mv $csv_file_name ${SORT}_outputs
          mv $pdf_file_name ${SORT}_outputs/pdfs
          mv replay.noise $noise_file_name
          mv replay.order $order_file_name
          mv $noise_file_name ${SORT}_outputs
          mv $order_file_name ${SORT}_outputs
        done
      done
    done
  done
done
