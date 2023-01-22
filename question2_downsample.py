import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from question_utils import set_seed, getLogger
from validation_lq.tinyface_helper import get_all_files

set_seed(0)
logger = getLogger(__name__, "INFO")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Downsample dataset')
    parser.add_argument('--root', default='data/custom_data_final')
    parser.add_argument('--add_noise', default=False, help="if set, add Gaussian noise", action="store_true")
    parser.add_argument('--add_blur', default=False, help="if set, blur images using Resize and Gaussian filter", action="store_true")
    args = parser.parse_args()

    assert(args.add_noise or args.add_blur), "At least add_noise or add_blur must be true"

    transform_ops = [
        transforms.Resize(28),
        transforms.Resize(112),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.05),
    ]
    postfix = "_lq"
    if not args.add_noise:
        transform_ops = transform_ops[:-1]
        postfix = "_lq_wo_noise"
    if not args.add_blur:
        transform_ops = transform_ops[-2:]
        postfix = "_noise"

    transform = transforms.Compose(transform_ops)

    dir_list = ["gallery", "probe", "distractor"]

    for current_dir in dir_list:
        current_root = os.path.join(args.root, current_dir)
        logger.info(f"Downsampling {current_root}")
        for filename in tqdm(get_all_files(current_root)):
            img = Image.open(filename)

            transformed_img = transform(img)
            transformed_img = transformed_img.permute(1,2,0).numpy()
            transformed_img = np.clip(transformed_img, 0., 1.)
            transformed_img = Image.fromarray((transformed_img * 255).astype(np.uint8))
            
            new_filename = filename.replace(current_root, current_root + postfix)
            os.makedirs(Path(new_filename).parent, exist_ok=True)
            transformed_img.save(new_filename)