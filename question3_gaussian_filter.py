import argparse
import os
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from question_utils import getLogger
from validation_lq.tinyface_helper import get_all_files

logger = getLogger(__name__, "INFO")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix dataset')
    parser.add_argument('--root', default='data/custom_data_final')
    args = parser.parse_args()

    transform = transforms.GaussianBlur(5)
    postfix = "_gaussian"

    dir_list = ["gallery_lq", "probe_lq", "distractor_lq"]

    for current_dir in dir_list:
        current_root = os.path.join(args.root, current_dir)
        logger.info(f"Fixing {current_root} using Gaussian smoothing")
        for filename in tqdm(get_all_files(current_root)):
            img = Image.open(filename)
            transformed_img = transform(img)

            new_filename = filename.replace(current_root, current_root + postfix)
            os.makedirs(Path(new_filename).parent, exist_ok=True)
            transformed_img.save(new_filename)