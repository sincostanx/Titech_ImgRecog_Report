#!/bin/bash

dir_list_face=( 
'data/custom_data_final/gallery_lq' 
'data/custom_data_final/probe_lq' 
'data/custom_data_final/distractor_lq'  
)

dir_list_denoise=( 
'../data/custom_data_final/gallery_lq' 
'../data/custom_data_final/probe_lq' 
'../data/custom_data_final/distractor_lq'  
)

face_hallucination_model_path='./SISN-Face-Hallucination/pretrained/FFHQ_256_X4.pt'
face_postfix='_face_hau'

denoising_postfix='_denoised'

# face hallucination
for dir in "${dir_list_face[@]}"; do
    python SISN-Face-Hallucination/test.py --model SISN --scale 4 --pretrain "$face_hallucination_model_path" --dataset_root "$dir" --save_root "$dir$face_postfix"
done

# image denoising
cd ./MPRNet
for dir in "${dir_list_denoise[@]}"; do
    python demo.py --task Denoising --input_dir "$dir" --result_dir "$dir$denoising_postfix"
done