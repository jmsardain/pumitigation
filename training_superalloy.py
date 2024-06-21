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

print("Python version:", sys.version)
print(tf.__version__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate NN model.')
    parser.add_argument('--train', action='store_const', const=True, default=False, help='Train NN  (default: False)')
    parser.add_argument('--outdir', type=str, default='', help='Directory with output is stored')
    parser.add_argument('--test', action='store_const', const=True, default=False, help='Test NN   (default: False)')
    parser.add_argument('--closure', action='store_const', const=True, default=False, help='Closure')
    
    return parser.parse_args()

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

def train_model(train_data, val_data, output_dir):
    columns_to_use = ['cluster_fracE', 'cluster_nCells', 'avgMu', 'nPrimVtx', 'clusterE', 'clusterEta', 'cluster_CENTER_MAG',
                      'cluster_FIRST_ENG_DENS', 'cluster_CENTER_LAMBDA', 'cluster_LATERAL',
                      'cluster_LONGITUDINAL', 'cluster_ISOLATION', 'cluster_SIGNIFICANCE',
                      'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS']

    X_train = train_data[columns_to_use]
    y_train = train_data['labels']
    X_val = val_data[columns_to_use]
    y_val = val_data['labels']

    model = build_model(X_train.shape[1])
    optimizer = Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

    model_json = model.to_json()
    with open(os.path.join(output_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(output_dir, "model_weights.h5"))
    print('Saved model architecture and weights.')

    plot_training_history(history, output_dir)

def evaluate_model(test_data, output_dir):
    columns_to_use = ['cluster_fracE', 'cluster_nCells', 'avgMu', 'nPrimVtx', 'clusterE', 'clusterEta', 'cluster_CENTER_MAG',
                      'cluster_FIRST_ENG_DENS', 'cluster_CENTER_LAMBDA', 'cluster_LATERAL',
                      'cluster_LONGITUDINAL', 'cluster_ISOLATION', 'cluster_SIGNIFICANCE',
                      'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS']

    X_test = test_data[columns_to_use]
    y_test = test_data['labels']

    with open(os.path.join(output_dir, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(output_dir, "model_weights.h5"))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    predictions = pd.DataFrame({
        'true_labels': y_test,
        'predicted_labels': y_pred,
        'predicted_probabilities': y_pred_proba
    })
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
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

    prob_df = pd.DataFrame(y_pred_proba, columns=['Probability of Pileup'])
    plt.figure(figsize=(10, 6))
    plt.hist(prob_df['Probability of Pileup'], bins=30, alpha=0.7, color='red', label='Probability of Pileup')
    plt.hist(1 - prob_df['Probability of Pileup'], bins=30, alpha=0.7, color='blue', label='Probability of No Pileup')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Predicted Probability Histograms')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'predicted_probability_histograms.png'))
    print('Saved predicted probability histograms plot.')

    df_weight = pd.DataFrame({
        'pred_proba': y_pred_proba,
        'labels': y_test
    })
    sig = df_weight[df_weight['labels'] == 1].pred_proba
    pu = df_weight[df_weight['labels'] == 0].pred_proba
    bins = np.linspace(0, 1, 100 + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(sig, bins=bins, label='sig', alpha=0.3)
    plt.hist(pu, bins=bins, label='pu', alpha=0.3, color='red')
    plt.yscale('log')
    plt.xlabel('NN Score')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Score Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_plot.png'))
    print('Saved score plot.')

def save_closure_data(train_data, output_dir):
    columns_to_use = ['cluster_fracE', 'cluster_nCells', 'avgMu', 'nPrimVtx', 'clusterE', 'clusterEta', 'cluster_CENTER_MAG',
                      'cluster_FIRST_ENG_DENS', 'cluster_CENTER_LAMBDA', 'cluster_LATERAL',
                      'cluster_LONGITUDINAL', 'cluster_ISOLATION', 'cluster_SIGNIFICANCE',
                      'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS']

    X_train = train_data[columns_to_use]
    y_train = train_data['labels']

    with open(os.path.join(output_dir, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(output_dir, "model_weights.h5"))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    y_pred = model.predict(X_train).flatten()

    closure_data = pd.DataFrame({
        'true_response': y_train,
        'pred_response': y_pred
    })
    closure_data.to_csv(os.path.join(output_dir, 'closure_data.csv'), index=False)

    X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    X_train_df.to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
    print('Saved closure data and training features as CSV.')

def main():
    args = parse_arguments()

    train_file_path = '/home/shfoster/calo-jad/calo-jad/MLClusterCalibration/training/transformed_train-1.csv'
    test_file_path = '/home/shfoster/calo-jad/calo-jad/MLClusterCalibration/training/transformed_test-1.csv'
    val_file_path = '/home/shfoster/calo-jad/calo-jad/MLClusterCalibration/training/transformed_val-1.csv'

    if args.train:
        train_data = pd.read_csv(train_file_path)
        val_data = pd.read_csv(val_file_path)
        train_model(train_data, val_data, args.outdir)

    if args.test:
        test_data = pd.read_csv(test_file_path)
        evaluate_model(test_data, args.outdir)

    if args.closure:
        train_data = pd.read_csv(train_file_path)
        save_closure_data(train_data, args.outdir)

if __name__ == "__main__":
    main()
