''' Read in csv data file of 1M calibration events
'''

#########################
### imports ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot as ur
########################
### MPL Settings ###

params = {
    'font.size': 14
}
plt.rcParams.update(params)

#########################

def make_all_plots(df, output_path):

    # for making plots
    with PdfPages(output_path) as pdf:
        # loop over field names
        for idx, key in enumerate(df):

            print(f"Accessing variable with name = {key} ({idx+1} / {df.shape[1]})")
            data = df[key].to_numpy()

            ##############
            # make plots #
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[5.5*3, 5.])

            # linear scale plot
            bins = 50
            _, bin_edges, _ = ax1.hist(data, bins=bins, histtype="step", density=False, label="linear")
            ax1.set_xlabel(key)
            ax1.set_ylabel("Frequency")

            if data.min() <= 0:
                data_upshifted = data + np.abs(data.min()) + 1e-30
                label = "log (upshifted)"
            else:
                data_upshifted = data
                label = "log"

            # log scale for y-axis
            ax2.hist(data_upshifted, bins=bin_edges, histtype="step", density=False, label=label)
            ax2.set_yscale("log")
            ax2.set_xlabel(key)
            ax2.legend(frameon=False, loc="upper right")
            ax2.set_ylabel("Frequency")

            # log scale for both axis
            bins_log = np.logspace(np.log10(data_upshifted.min()), np.log10(data_upshifted.max()), len(bin_edges)-1)
            ax3.hist(data_upshifted, bins=bins_log, histtype="step", density=False, label=label)
            ax3.set_yscale("log")
            ax3.set_xscale("log")
            ax3.set_xlabel(key)
            ax3.legend(frameon=False, loc="upper right")
            ax3.set_ylabel("Frequency")
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saving all figures into {output_path}")


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

#########################

def main():

    ################
    ### params ###

    file_path = "all_info_df"
    filename = ur.open('/home/jmsardain/JetCalib/PUMitigation/latest/MakeROOT/output.root')["ClusterTree"]
    df = filename.arrays(library="pd")

    df = df[df["cluster_ENG_CALIB_TOT"]>0.]
    df = df[df["clusterE"]>0.]
    df = df[df["cluster_CENTER_LAMBDA"]>0.]
    df = df[df["cluster_FIRST_ENG_DENS"]>0.]
    df = df[df["cluster_SECOND_TIME"]>0.]
    df = df[df["cluster_SIGNIFICANCE"]>0.]

    resp = np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values )
    df["response"] = resp
    df = df[df["response"]>0.1]

    print(df.columns)

    column_names = ['response', 'clusterECalib', 'cluster_ENG_CALIB_TOT', 'jetCnt',
                    'clusterE', 'clusterEta', 'clusterPhi', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
                    'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
                    'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
                    'nPrimVtx', 'avgMu',
                ]

    before = df
    df = df[column_names]
    print(df.columns)

    output_path_figures_before_preprocessing = "fig.pdf"
    output_path_figures_after_preprocessing = "fig2.pdf"
    output_path_data = "data/" + file_path + ".npy"
    # output_path_data = "data/"
    save = True
    scales_txt_file =  "scales.txt"

    print(df)



    #####################
    ### preprocessing ###
    print("-"*100)
    print("Make preprocessing...")

    scales = {}
    # log-preprocessing
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS", "cluster_SECOND_TIME", "cluster_SIGNIFICANCE"]
    for field_name in field_names:
        x, minimum, epsilon = apply_save_log(df[field_name])
        x, mean, std = normalize(x)
        scales[field_name] = ("SaveLog / Normalize", minimum, epsilon, mean, std)
        df[field_name] = x

    # just normalizing
    field_names = ["clusterEta", "clusterPhi", "cluster_CENTER_MAG", "nPrimVtx", "avgMu"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        scales[field_name] = ("Normalize", mean, std)
        df[field_name] = x

    # params between [0, 1]
    # we could also just shift?
    field_names = ["cluster_ENG_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL", "cluster_PTD", "cluster_ISOLATION"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        scales[field_name] = ("Normalize", mean, std)
        df[field_name] = x

    # special preprocessing
    field_name = "cluster_time"
    x = df[field_name]
    x = np.abs(x)**(1./3.) * np.sign(x)
    x, mean, std = normalize(x)
    #x = np.tanh(x)
    #x = x + np.abs(x.min()) + 1
    #x = apply_save_log(df[field_name])
    #x = normalize(x)
    scales[field_name] = ("Sqrt3", mean, std)
    df[field_name] = x

    # no preprocessing
    #files_names = ["r_e_calculated"]

    ######################

    print("-"*100)
    print("Make plots after preprocessing..,")

    # plots after preprocessing
    # make_all_plots(df, output_path_figures_after_preprocessing)

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

if __name__ == "__main__":
    main()
