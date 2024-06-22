import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate NN model.')
    parser.add_argument('--train', action='store_const', const=True, default=False, help='Train NN  (default: False)')
    parser.add_argument('--test', action='store_const', const=True, default=False, help='Test NN   (default: False)')
    parser.add_argument('--outdir', type=str, default='', help='Directory with output is stored')
    parser.add_argument('--lr',   dest='lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--epochs',   dest='epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size',   dest='batch_size', type=int, default=64, help='batch size')

    return parser.parse_args()


def main():
    args = parse_arguments()

    train_file_path = '/home/shfoster/calo-jad/calo-jad/MLClusterCalibration/training/transformed_train-1.csv'
    test_file_path = '/home/shfoster/calo-jad/calo-jad/MLClusterCalibration/training/transformed_test-1.csv'
    val_file_path = '/home/shfoster/calo-jad/calo-jad/MLClusterCalibration/training/transformed_val-1.csv'

    columns_to_use = ['cluster_fracE', 'cluster_nCells', 'avgMu', 'nPrimVtx', 'clusterE', 'clusterEta', 'cluster_CENTER_MAG',
                      'cluster_FIRST_ENG_DENS', 'cluster_CENTER_LAMBDA', 'cluster_LATERAL',
                      'cluster_LONGITUDINAL', 'cluster_ISOLATION', 'cluster_SIGNIFICANCE',
                      'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS']


    if args.train:
        ## Get training data
        train_data = pd.read_csv(train_file_path)
        X_train = train_data[columns_to_use]
        y_train = train_data['labels']
        ## Get validation data
        val_data = pd.read_csv(val_file_path)
        X_val   = val_data[columns_to_use]
        y_val   = val_data['labels']

        ## Build model
        model = build_model(X_train.shape[1])
        ## Train
        train_data, val_data, model, lr=0.00001, output_dir=''):
        history = train_model(train_data, val_data, model, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.outdir)
        plot_training_history(history, output_dir)

    if args.test:
        ## Get testing data
        test_data = pd.read_csv(test_file_path)
        X_test  = test_data[columns_to_use]
        y_test  = test_data['labels']
        ## Here, one should add the untransformed data set as well so that you retrieve jet info correctly

        ## Get saved model
        with open(os.path.join(args.outdir, "model.json"), "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(args.outdir, "model_weights.h5"))

        score, pred_labels = evaluate_model(test_data, model, output_dir=args.outdir)
        print_eval_metrics(y_test, pred_labels)
        plot_ROC_curve(y_test, score, output_dir=args.outdir)
        plot_score(y_test, score)

if __name__ == "__main__":
    main()
