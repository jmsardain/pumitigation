import numpy as np
import pandas as pd
# import torch

from utils import *
from models import *


def normalize(x):
    mean, std = np.mean(x), np.std(x)
    out =  (x - mean) / std
    return out, mean, std

def apply_save_log(x):

    #########
    epsilon = 1e-10
    #########

    minimum = x.min()
    if x.min() <= 0:
        x = x - x.min() + epsilon
    else:
        minimum = 0
        epsilon = 0

    return np.log(x), minimum, epsilon

# Main function.
def main():
    filename = '/home/jmsardain/JetCalib/PUMitigation/fracdata.csv'
    df = pd.read_csv(filename, sep=' ')


    column_names = ['clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
                'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
                'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
                'nPrimVtx', 'avgMu']

    df['labels'] = ((df['cluster_ENG_CALIB_TOT'] == 0) & (df['clusterE'] > 0)).astype(int)


    before = df
    df = df[column_names]
    df['labels'] = before['labels']
    
    file_path = "all_info_df"
    output_path_figures_before_preprocessing = "fig.pdf"
    output_path_figures_after_preprocessing = "fig2.pdf"
    output_path_data = "data/" + file_path + ".npy"

    save = True
    scales_txt_file =  output_path_data[:-4] + "_scales.txt"

    scales = {}
    # log-preprocessing
    field_name = "cluster_time"
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS", "cluster_SECOND_TIME", "cluster_SIGNIFICANCE"]
    for field_name in field_names:
        x, minimum, epsilon = apply_save_log(df[field_name])
        x, mean, std = normalize(x)
        df[field_name] = x

    # just normalizing
    field_names = ["clusterEta", "cluster_CENTER_MAG", "nPrimVtx", "avgMu"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        df[field_name] = x

    # params between [0, 1]
    # we could also just shift?
    field_names = ["cluster_ENG_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL", "cluster_PTD", "cluster_ISOLATION"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        df[field_name] = x

    # special preprocessing
    field_name = "cluster_time"
    x = df[field_name]
    x = np.abs(x)**(1./3.) * np.sign(x)
    x, mean, std = normalize(x)

    df[field_name] = x

    if save:
        brr = before.to_numpy()
        arr = df.to_numpy()
        print(arr.shape)
        print(scales)
        np.save(output_path_data, arr)
        print("Saved as {output_path_data}")
        with open(scales_txt_file, "w") as f:
            f.write(str(scales))


        n = len(arr)
        ntrain = int(n * 0.6)
        ntest = int(n * 0.2)

        train = arr[:ntrain]
        val  = arr[ntrain+ntest:]
        test  = arr[ntrain:ntrain+ntest]
        ## clusterE / response = true energy should be higher than 0.3


        test_before= brr[ntrain:ntrain+ntest]

        np.save("data/all_info_df_train.npy", train)
        np.save("data/all_info_df_test.npy", test)
        np.save("data/all_info_df_val.npy", val)
        np.save("data/all_info_df_test_before.npy", test_before)


    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
