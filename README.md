# ART.T551 (Final Report)

This repository contains codes to reproduce experiments presented in [the final report](https://drive.google.com/file/d/1dsRsl-j0uV9iWdGbOPw7SfHvkBT_qMSR/view?usp=sharing) of Image and Video Recognition class (ART.T551) in 2022, written by Worameth Chinchuthakun (22M51878). Please clone this repository directly instead of cloning the original repositories separately because we modified some parts of their codes.

## Installation
Please confirm your CUDA version before proceed.

```bash
conda create --name adaface
conda activate adaface
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Task 1: Replicating state-of-the-art results
Here, we describe steps for reproducing results of ResNet100 (trained with MS1MV2) on TinyFace dataset.

- Download the pretrained model from [here](https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view). Place it in directory ```./pretrained```.
- Download evaluation split of TinyFace from [here](https://qmul-tinyface.github.io/). Extract the zip file under ```<data_root>```.
- Download preprocessed (i.e. aligned and resized) TinyFace dataset from by complething this [form](https://docs.google.com/forms/d/e/1FAIpQLSc3zWLSUf3DH6F0LXhjecolZYt63EMLkGX-2sayz2WbfhbDcA/viewform). The download link will appear after that. Then, extract the zip file under ```<data_root>```.
- Run the following script
```bash
cd ./validation_lq
python validate_tinyface.py --data_root <data_root> --model_name ir101_ms1mv2
```

Evaluation results are available as a csv file at ```./validation_lq/tinyface_result```.

## Task 2: Evaluating on custom dataset
Here, we describe steps for reproducing results of ResNet100 (trained with MS1MV2) on MWITS-20 (our original dataset). Do not forget to download the pretrained model as described in Task 1. To protect privacy of people in images, we do not share the original images. Still, the pre-processing code is available at ```question2_preprocess.py``` as a reference.

- Download the preprocessed MWITS-20 from [here](https://drive.google.com/file/d/1g6l2NwJxHOoVT54WYNhN0ZwZb-dkz7n3/view?usp=sharing). Extract the zip file in directory ```./data/```.
- Downsample and add Gaussian noise to the original dataset by running the following script.
```bash
bash question2_downsample.sh
```
Confirm that there are directory ```./data/custom_data_final/{x}_{y}``` where ```x = [gallery, probe, distractor]``` and ```x = [lq_wo_noise, noise, lq]```.
- Evaluate performance by running the following script.
```bash
bash question2_evaluate.sh
```

Evaluation results are available as csv files in ```./experiment_result```. Visuailizations of correct and incorrect matching of some cases are available in ```./visualize_matched``` and ```./visualize_unmatched```, respectively.

## Task 3: Improving evaluation results on custom dataset
- Download SISN pretrained model (```FFHQ_256_X4.pt```) from [here](https://drive.google.com/drive/folders/1CX1ERx6A3C8LAe6Mj9LtQ9rUhuuHZXil). Place it in directory ```./SISN-Face-Hallucination/pretrained```.
- Download MPRNet (Denoising) pretrained model from [here](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view). Place it in directory ```./MPRNet/Denoising/pretrained_models```.
- Run the following script to restore corrupted images using Gaussian filter, SISN, and MPRNet.
```bash
bash question3_preprocess.sh
```
Confirm that there are directory ```./data/custom_data_final/{x}_{y}``` where ```x = [gallery, probe, distractor]``` and ```x = [lq_gaussian, lq_face_hau, lq_denoised]```.
- Run the following script to evaluate performance
```bash
bash question3_evaluate.sh
```
Evaluation results are available as csv files in ```./experiment_result```.

## Others
- ```final_report_visualization.ipynb``` contains code for generating visualization used in the final report.

## External links
- [AdaFace](https://github.com/mk-minchul/AdaFace)
- [SISN](https://github.com/mdswyz/SISN-Face-Hallucination#dataset)
- [MPRNet](https://github.com/swz30/MPRNet)