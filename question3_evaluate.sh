#!/bin/bash

# compare_res=( 'lq_gaussian' 'lq_face_hau' 'lq_denoised' )
compare_res=( 'lq_gaussian' )

for res in "${compare_res[@]}"; do
    supported_res=( 'hq' "$res" )
    for i in "${supported_res[@]}"; do
        for j in "${supported_res[@]}"; do
            for k in "${supported_res[@]}"; do
                python question2_evaluate.py --gallery_resolution "$i" --probe_resolution "$j" --distractor_resolution "$k"
            done
        done
    done
    python question2_aggregate_result.py --res1 hq --res2 "$res"
done