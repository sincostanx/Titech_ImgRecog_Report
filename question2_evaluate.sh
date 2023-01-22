#!/bin/bash

compare_res=( 'lq_wo_noise' 'noise' 'lq' )
output_hq_flag=true

for res in "${compare_res[@]}"; do
    supported_res=( 'hq' "$res" )
    for i in "${supported_res[@]}"; do
        for j in "${supported_res[@]}"; do
            for k in "${supported_res[@]}"; do
                visualize='none'
                if [[ "$i" = 'hq' && "$j" = 'hq' && "$k" = 'hq' && "$output_hq_flag" = true ]]; then
                    visualize='all'
                    output_hq_flag=false
                elif [[ "$i" = 'hq' && "$j" = 'lq' && "$k" = 'hq' ]]; then
                    visualize='incorrect'
                fi
                python question2_evaluate.py --gallery_resolution "$i" --probe_resolution "$j" --distractor_resolution "$k" --visualize "$visualize"
            done
        done
    done
    python question2_aggregate_result.py --res1 hq --res2 "$res"
done