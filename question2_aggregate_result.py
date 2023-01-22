import itertools
import pandas as pd
import os
import numpy as np
import argparse

def generate_compare_table(res1, res2):
    # extract evaluation results
    root = "custom_dataset_result/ir101_ms1mv2/fusion_pre_norm_vector_add"
    options = [res1, res2]
    df = []
    for i, j, k in itertools.product(options, options, options):
        log_name = f'result_gallery-{i}_probe-{j}_distractor-{k}.csv'
        filename = os.path.join(root, log_name)
        temp_df = pd.read_csv(filename)
        scores = temp_df["values"].to_numpy()
        result = {
            "Gallery": i,
            "Probe": j,
            "Distractor": k,
            "Rank-1": np.round(scores[0] * 100, 2),
            "Rank-5": np.round(scores[1] * 100, 2),
            "Rank-20": np.round(scores[2] * 100, 2),
        }
        df.append(result)

    # aggregate result
    df = pd.DataFrame.from_dict(df)
    df["Rank-1 difference"] = df["Rank-1"] - df["Rank-1"][0]
    df["Rank-1 difference"] = df["Rank-1 difference"].apply(lambda x: np.round(x, 2))

    # save compare table
    save_dir = "./experiment_result"
    save_filename = f"compare_table_{res1}_{res2}.csv"
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, save_filename), index=False)

if __name__ == '__main__':

    supported_options = ('hq', 'lq', 'lq_wo_noise', 'noise', 'lq_gaussian', 'lq_face_hau', 'lq_denoised')
    parser = argparse.ArgumentParser(description='tinyface')

    parser.add_argument('--res1', type=str, required=True, choices=supported_options)
    parser.add_argument('--res2', type=str, required=True, choices=supported_options)

    args = parser.parse_args()

    generate_compare_table(args.res1, args.res2)