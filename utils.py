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

def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model


def train_model(train_data, val_data, model, lr=0.00001, epochs=300, batch_size=64, output_dir=''):

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    model_json = model.to_json()

    with open(os.path.join(output_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(output_dir, "model_weights.h5"))
    print('Saved model architecture and weights.')
    return history

def evaluate_model(test_data, model, output_dir=''):

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    return y_pred_proba, y_pred


def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_validation_metrics.png'))
    print('Saved training and validation metrics plot.')



def print_eval_metrics(y_test, pred_labels):
    precision = precision_score(y_test, pred_labels)
    recall = recall_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    conf_matrix = confusion_matrix(y_test, pred_labels)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

def plot_ROC_curve(y_test, score, output_dir=''):

    fpr, tpr, _ = roc_curve(y_test, score)
    roc_auc = roc_auc_score(y_test, score)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    print('Saved ROC curve plot.')

def plot_score(y_test, score):
    y_test = np.array(y_test)
    score = np.array(score)
    pu_scores  = score[y_test == 0]
    sig_scores = score[y_test == 1]

    bins = np.linspace(0, 1, 100 + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(sig_scores, bins=bins, label='sig', alpha=0.3)
    plt.hist(pu_scores, bins=bins, label='pu', alpha=0.3, color='red')
    plt.yscale('log')
    plt.xlabel('NN Score')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Score Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_plot.png'))
    print('Saved score plot.')
