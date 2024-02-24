import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import uproot as ur
import datetime, os
import tensorflow as tf
from sklearn.preprocessing import  QuantileTransformer, StandardScaler
from sklearn.utils import class_weight

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn import ensemble

from keras.backend import sigmoid
import uproot
from keras import regularizers
import keras.backend as K
def swish(x, beta = 1):
    return 2*(x * sigmoid(beta * x))

def tanhPlusOne(x):
    return 2*(tf.keras.activations.tanh(x) + 1)
    # return 6*(tf.keras.activations.tanh(x))

from keras.utils import get_custom_objects
from keras.layers import Activation
from keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Below in place of swish you can take any custom key for the name
get_custom_objects().update({'swish': swish})
get_custom_objects().update({'tanhPlusOne': tanhPlusOne})

# Register custom objects# custom_objects = {'SwishActivation': SwishActivation, 'TanhPlusOneActivation': TanhPlusOneActivation}
# tf.keras.utils.get_custom_objects().update(custom_objects)

parser = argparse.ArgumentParser(description='Perform signal injection test.')
parser.add_argument('--train',    dest='train',    action='store_const', const=True, default=False, help='Train NN  (default: False)')
parser.add_argument('--retrain',  dest='retrain',  action='store_const', const=True, default=False, help='Retrain NN  (default: False)')
parser.add_argument('--test',     dest='test',     action='store_const', const=True, default=False, help='Test NN   (default: False)')
parser.add_argument('--closure',  dest='closure',  action='store_const', const=True, default=False, help='Closure')
parser.add_argument('--weight',   dest='weight',   type=int, default=0, help='Weight: 0: no weight 1: response, 2: response wider, 3: energy, 4: log energy')
parser.add_argument('--outdir',   dest='outdir',   type=str, default='', help='Directory with output is stored')


args = parser.parse_args()

def lgk_loss_function(y_true, y_pred): ## https://arxiv.org/pdf/1910.03773.pdf
    alpha = tf.constant(0.05)
    bandwith = tf.constant(0.1)
    pi = tf.constant(math.pi)
    ## LGK (h and alpha are hyperparameters)
    norm = -1/(bandwith*tf.math.sqrt(2*pi))
    gaussian_kernel  = norm * tf.math.exp( -(y_pred/y_true - 1)**2 / (2*(bandwith**2)))
    leakiness = alpha*tf.math.abs(y_pred/y_true - 1)
    lgk_loss = gaussian_kernel + leakiness
    loss = lgk_loss
    return loss

def lgk_loss_function_1(y_true, y_pred): ## https://arxiv.org/pdf/1910.03773.pdf
    alpha = tf.constant(0.05)
    bandwith = tf.constant(0.1)
    pi = tf.constant(math.pi)
    ## LGK (h and alpha are hyperparameters)
    norm = -1/(bandwith*tf.math.sqrt(2*pi))
    gaussian_kernel  = norm * tf.math.exp(  -(y_pred/y_true - 1)**2 / (2*(bandwith**2)))
    lgk_loss = gaussian_kernel
    loss = lgk_loss
    return loss


def build_and_compile_model(X_train, lr):
    model = keras.Sequential([layers.Flatten(input_shape=(X_train.shape[1],)),
                                             layers.Dense(256, activation='swish'),
                                             layers.Dense(128, activation='swish'),
                                             layers.Dense(64,  activation='swish'),
                                             layers.Dense(8,   activation='swish'),
                                             layers.Dense(1,   activation='tanhPlusOne')])
    model.compile(loss=lgk_loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=[])
    return model


# Main function.
def main():

    dir_path = "ckpts/"
    try:
        os.system("mkdir {}".format(dir_path))
    except ImportError:
        print("{} already exists".format(dir_path))
    pass

    # train dataset
    dataset_train = np.load('data/all_info_df_train.npy')
    x_train = dataset_train[:, 3:]
    y_train = dataset_train[:, 0]
    # w_train = dataset_train[:, 4] ## 1: response, 2: response wider, 3: energy, 4: log energy
    data_train = np.concatenate([x_train, y_train[:, None]], axis=-1)
    print(f"Training dataset size {y_train.shape[0]}")

    # val dataset
    dataset_val = np.load('data/all_info_df_val.npy')
    x_val = dataset_val[:, 3:]
    y_val = dataset_val[:, 0]
    data_val = np.concatenate([x_val, y_val[:, None]], axis=-1)
    print(f"Validation dataset size {y_val.shape[0]}")

    # test dataset
    dataset_test = np.load('data/all_info_df_test.npy')
    x_test = dataset_test[:, 3:]
    y_test = dataset_test[:, 0]
    data_test = np.concatenate([x_test, y_test[:, None]], axis=-1)
    print(f"Test dataset size {y_test.shape[0]}")
    

    if args.train:
        nepochs = 5
        with tf.device('/GPU:0'):
            dnn_model = build_and_compile_model(x_train, lr=0.0001)
            # print(dnn_model.summary())
            history = dnn_model.fit( x_train, y_train, validation_split=0.35, epochs=nepochs, batch_size=4096) # batch_size=1024

        metrics = pd.DataFrame({"Train_Loss":history.history['loss'],"Val_Loss":history.history['val_loss']})
        metrics.to_csv('Losses_train_leakiness.csv', index = False)
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + '/Losses_train_leakiness.png')
        plt.clf()
        dnn_model.save(dir_path+"/model_leakiness.h5")
        y_pred = dnn_model.predict(x_test).flatten()
        ### start saving output
        np.save(dir_path + '/trueResponse.npy', y_test)
        np.save(dir_path + '/predResponse.npy', y_pred)
        np.save(dir_path + '/x_test.npy',       x_test)



    
    if args.retrain:
        nepochs = 5
        with tf.device('/GPU:0'):
            custom_objects = {'swish': swish, 'tanhPlusOne': tanhPlusOne}
            tf.keras.utils.get_custom_objects().update(custom_objects)
            dnn_model = tf.keras.models.load_model(dir_path+"/model_leakiness.h5", compile=False, custom_objects={'swish': swish, 'tanhPlusOne': tanhPlusOne})
            # dnn_model = tf.keras.models.load_model(dir_path+"/model_leakiness.h5", compile=False)
            dnn_model.compile(loss=lgk_loss_function_1, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), weighted_metrics=[])
            history = dnn_model.fit( x_train, y_train, validation_split=0.35, epochs=nepochs, batch_size=4096) # batch_size=1024


        metrics = pd.DataFrame({"Train_Loss":history.history['loss'],"Val_Loss":history.history['val_loss']})
        metrics.to_csv('Losses_final.csv', index = False)

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + '/Losses_final.png')
        plt.clf()
        dnn_model.save(dir_path+"/model.h5")

    ## testing
    if args.test:
        custom_objects = {'swish': swish, 'tanhPlusOne': tanhPlusOne}
        tf.keras.utils.get_custom_objects().update(custom_objects)
        dnn_model = tf.keras.models.load_model(dir_path+"/model.h5", compile=False, custom_objects=custom_objects)
        if dnn_model == None:
            return
        y_pred = dnn_model.predict(x_test).flatten()
        ### start saving output
        np.save(dir_path + '/trueResponse.npy', y_test)
        np.save(dir_path + '/predResponse.npy', y_pred)
        np.save(dir_path + '/x_test.npy',       x_test)

    if args.closure:
        dnn_model = tf.keras.models.load_model(dir_path+"/model.h5", compile=False)
        if dnn_model == None:
            return
        y_pred = dnn_model.predict(x_train).flatten()
        ### start saving output
        np.save(dir_path + '/trueResponse_closure.npy', y_train)
        np.save(dir_path + '/predResponse_closure.npy', y_pred)
        np.save(dir_path + '/x_train.npy',       x_train)

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
