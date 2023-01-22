#!/bin/bash

supported_res=( "hq" "lq_face_hau" )

for i in "${supported_res[@]}"; do
    for j in "${supported_res[@]}"; do
        for k in "${supported_res[@]}"; do
            python question2_evaluate.py --gallery_resolution "$i" --probe_resolution "$j" --distractor_resolution "$k"
        done
    done
done