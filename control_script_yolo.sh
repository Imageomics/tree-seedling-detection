#!/bin/bash
mkdir -p logs

epochs=(2000)
batch_sizes=(32 64) 
modalities=('rgb')


# Variable to hold the ID of the last submitted job
last_job_id=""

#Single-Modal AWIR Experiments

for batch_size in ${batch_sizes[@]}; do
    for modality in ${modalities[@]}; do
        for epoch in ${epochs[@]}; do

            echo "Preparing to run with batch_size=${batch_size}, epoch=${epoch}, modality=${modality}"

            # Check if there's a job to depend on
            if [[ -z $last_job_id ]]; then
                # No dependency
                last_job_id=$(sbatch --parsable --job-name="yolo_${modality}" \
                          --output=logs/yolo${batch_size}_${modality}_${epoch}_%j.out \
                          run_yolo.sh $batch_size $modality $epoch)
            else
                # Submit with dependency on the completion of the last job
                last_job_id=$(sbatch --parsable --dependency=afterok:$last_job_id \
                          --job-name="yolo_${modality}" \
                          --output=logs/yolo_${batch_size}_${modality}_${epoch}_%j.out \
                          run_yolo.sh $batch_size $modality $epoch)
            fi
            echo "Submitted job $last_job_id with batch_size=${batch_size}, epoch=${epoch}, modality=${modality}"


        done
    done
done
