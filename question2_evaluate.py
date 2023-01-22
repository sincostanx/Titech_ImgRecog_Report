import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from question_utils import getLogger
from validation_lq.data_utils import prepare_dataloader
from validation_lq.tinyface_helper import get_all_files
from validation_lq.validate_tinyface import (DIR_FAR, infer, inner_product,
                                             load_pretrained_model, str2bool)

logger = getLogger(__name__, "INFO")

def visualize_incorrect(root, gallery_res, probe_res, distract_res):
    # load prediction result
    filename = f'result_gallery-{gallery_res}_probe-{probe_res}_distractor-{distract_res}.pickle'
    filename = os.path.join(root, filename)
    with (open(filename, "rb")) as f:
        custom_test = pickle.load(f)

    x = np.vectorize(custom_test.get_label)(custom_test.gallery_paths)
    y = custom_test.gallery_paths
    actual_match_dict = dict(zip(x, y))

    output_dir = f"./visualize_unmatched/result_gallery-{gallery_res}_probe-{probe_res}_distractor-{distract_res}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Visualize directory: {output_dir}")

    countx = 0
    for probe_path, gallery_path in tqdm(custom_test.unmatched_list):

        fig, ax = plt.subplots(1, 3, figsize=(7, 10))
        ax = ax.ravel()

        probe_label = str(custom_test.get_label(probe_path))
        try:
            match_label = str(custom_test.get_label(gallery_path))
        except ValueError:
            match_label = "distractor"
        
        img = Image.open(probe_path)
        img = np.asarray(img)
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[0].set_title(f"Ground truth = {probe_label}")

        img = Image.open(gallery_path)
        img = np.asarray(img)
        ax[1].imshow(img)
        ax[1].axis("off")
        ax[1].set_title(f"Matched = {match_label}")

        img = Image.open(actual_match_dict[int(probe_label)])
        img = np.asarray(img)
        ax[2].imshow(img)
        ax[2].axis("off")
        ax[2].set_title(f"Correct match")
        
        fig.savefig(os.path.join(output_dir, f"{countx}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        countx += 1

def visualize_correct(root, gallery_res, probe_res, distract_res):
    # load prediction result
    filename = f'result_gallery-{gallery_res}_probe-{probe_res}_distractor-{distract_res}.pickle'
    filename = os.path.join(root, filename)
    with (open(filename, "rb")) as f:
        custom_test = pickle.load(f)

    x = np.vectorize(custom_test.get_label)(custom_test.gallery_paths)
    y = custom_test.gallery_paths
    actual_match_dict = dict(zip(x, y))

    output_dir = f"./visualize_matched/result_gallery-{gallery_res}_probe-{probe_res}_distractor-{distract_res}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Visualize directory: {output_dir}")

    countx = 0
    for probe_path, gallery_path in tqdm(custom_test.matched_list):

        fig, ax = plt.subplots(1, 2, figsize=(7, 7))
        ax = ax.ravel()

        probe_label = str(custom_test.get_label(probe_path))
        match_label = str(custom_test.get_label(gallery_path))
        
        img = Image.open(probe_path)
        img = np.asarray(img)
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[0].set_title(f"Ground truth = {probe_label}")

        img = Image.open(gallery_path)
        img = np.asarray(img)
        ax[1].imshow(img)
        ax[1].axis("off")
        ax[1].set_title(f"Matched = {match_label}")
        
        fig.savefig(os.path.join(output_dir, f"{countx}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        countx += 1

class CustomDatasetTest:
    def __init__(self, root, gallery_res, probe_res, distract_res):

        self.root = root

        # get image paths
        probe_dir = "probe" if probe_res == "hq" else f"probe_{probe_res}"
        gallery_dir = "gallery" if gallery_res == "hq" else f"gallery_{gallery_res}"
        distract_dir = "distractor" if distract_res == "hq" else f"distractor_{distract_res}"

        logger.info(f"Loading probe from {probe_dir}")
        logger.info(f"Loading gallery from {gallery_dir}")
        logger.info(f"Loading distractor from {distract_dir}")

        self.probe_paths = sorted(np.array(get_all_files(os.path.join(root, probe_dir))))
        self.gallery_paths = sorted(np.array(get_all_files(os.path.join(root, gallery_dir))))
        self.distractor_paths = sorted(np.array(get_all_files(os.path.join(root, distract_dir))))
        self.img_paths = np.concatenate([self.probe_paths,self.gallery_paths, self.distractor_paths])

        logger.info(f"Found {len(self.img_paths)} test images")
        logger.info(f"Probe images = {len(self.probe_paths)}")
        logger.info(f"Gallery (match) images = {len(self.gallery_paths)}")
        logger.info(f"Gallery (distraction) images = {len(self.distractor_paths)}")

        # get indices for images
        indices = np.array(list(range(len(self.img_paths))))
        self.indices_probe = indices[:len(self.probe_paths)]
        self.indices_match = indices[len(self.probe_paths):len(self.probe_paths) + len(self.gallery_paths)]
        self.indices_distractor = indices[len(self.probe_paths) + len(self.gallery_paths):]

        # get label for images
        self.labels_probe = np.vectorize(self.get_label)(self.probe_paths)
        self.labels_match = np.vectorize(self.get_label)(self.gallery_paths)
        self.labels_distractor = np.full_like(self.distractor_paths, -100, dtype=int)

        self.indices_gallery = np.concatenate([self.indices_match, self.indices_distractor])
        self.labels_gallery = np.concatenate([self.labels_match, self.labels_distractor])

    def get_label(self, image_path):
        if "probe" in image_path:
            dir_name = str(Path(image_path).parent.name) # e.g. './data/custom_data_final/probe/1_pang/1.png'
            label = int(dir_name.split("_")[0]) # e.g. ['1', 'pang']
        else:
            basename = os.path.basename(image_path)
            label = int(basename.split("_")[0]) # e.g. ['1', 'pang']
        
        return label

    def test_identification(self, features, ranks=[1,5,20]):
        # calculate similarity score
        feat_probe = features[self.indices_probe]
        feat_gallery = features[self.indices_gallery]
        compare_func = inner_product
        score_mat = compare_func(feat_probe, feat_gallery)

        # generate label matrix
        label_mat = np.equal.outer(self.labels_probe, self.labels_gallery)

        # evaluation and generate a list of probe that matched with a gallery (rank 1)
        results, _, _, _, _, _, _, sort_idx_mat_m = DIR_FAR(score_mat, label_mat, ranks, get_false_indices=True)
        self.sort_idx_mat_m = sort_idx_mat_m
        self.generate_matching_list()

        return results

    def generate_matching_list(self):
        self.gallery_img_paths = self.img_paths[self.indices_gallery]

        self.unmatched_list = []
        self.matched_list = []
        for i in range(self.sort_idx_mat_m.shape[0]):
            pred_identity = self.labels_gallery[self.sort_idx_mat_m[i, -1]]
            gt_identity = self.labels_probe[i]

            probe_path = img_paths[i]
            match_path = self.gallery_img_paths[self.sort_idx_mat_m[i, -1]]
            if pred_identity != gt_identity:
                self.unmatched_list.append((probe_path, match_path))
            else:
                self.matched_list.append((probe_path, match_path))


if __name__ == '__main__':

    supported_options = ('hq', 'lq', 'lq_wo_noise', 'noise', 'lq_gaussian', 'lq_face_hau', 'lq_denoised')
    parser = argparse.ArgumentParser(description='tinyface')

    parser.add_argument('--data_root', default='data/custom_data_final')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--model_name', type=str, default='ir101_ms1mv2')
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--fusion_method', type=str, default='pre_norm_vector_add', choices=('average', 'norm_weighted_avg', 'pre_norm_vector_add', 'concat', 'faceness_score'))
    parser.add_argument('--gallery_resolution', type=str, required=True, choices=supported_options)
    parser.add_argument('--probe_resolution', type=str, required=True, choices=supported_options)
    parser.add_argument('--distractor_resolution', type=str, required=True, choices=supported_options)
    parser.add_argument('--visualize', type=str, default='none', choices=('none', 'correct', 'incorrect', 'all'))
    
    args = parser.parse_args()

    # prepare test data
    params = {
        "root": args.data_root,
        "gallery_res": args.gallery_resolution,
        "probe_res": args.probe_resolution,
        "distract_res": args.distractor_resolution,
    }
    custom_test = CustomDatasetTest(**params)
    img_paths = custom_test.img_paths
    dataloader = prepare_dataloader(img_paths, args.batch_size, num_workers=16)

    # load model
    adaface_models = {
        'ir50': ["./pretrained/adaface_ir50_ms1mv2.ckpt", 'ir_50'],
        'ir101_ms1mv2': ["./pretrained/adaface_ir101_ms1mv2.ckpt", 'ir_101'],
        'ir101_ms1mv3': ["./pretrained/adaface_ir101_ms1mv3.ckpt", 'ir_101'],
        'ir101_webface4m': ["./pretrained/adaface_ir101_webface4m.ckpt", 'ir_101'],
        'ir101_webface12m': ["./pretrained/adaface_ir101_webface12m.ckpt", 'ir_101'],
    }
    assert args.model_name in adaface_models

    model = load_pretrained_model(args.model_name, adaface_models)
    model.to('cuda:{}'.format(args.gpu))

    # # inference
    features, norms = infer(model, dataloader, use_flip_test=args.use_flip_test, fusion_method=args.fusion_method)
    results = custom_test.test_identification(features, ranks=[1, 5, 20])
    logger.info(results)

    # # save output
    save_path = os.path.join('./custom_dataset_result', args.model_name, "fusion_{}".format(args.fusion_method))
    os.makedirs(save_path, exist_ok=True)

    log_filename = f'result_gallery-{args.gallery_resolution}_probe-{args.probe_resolution}_distractor-{args.distractor_resolution}.csv'
    log_path = os.path.join(save_path, log_filename)
    logger.info('Output path: {}'.format(log_path))

    output_df = pd.DataFrame({'rank':[1,5,20], 'values':results})
    output_df.to_csv(log_path, index=False)
    with open(log_path.replace(".csv", ".pickle"), 'wb') as f:
        pickle.dump(custom_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    # visualize incorrect predictions
    params["root"] = save_path
    if args.visualize in ["incorrect", "all"]: visualize_incorrect(**params)
    if args.visualize in ["correct", "all"]: visualize_correct(**params)
    
