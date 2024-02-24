import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import TensorDataset, DataLoader
import argparse
import glob
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Perform signal injection test.')
parser.add_argument('--train',    dest='train',    action='store_const', const=True, default=False, help='Train NN  (default: False)')
parser.add_argument('--retrain',  dest='retrain',  action='store_const', const=True, default=False, help='Retrain NN  (default: False)')
parser.add_argument('--test',     dest='test',     action='store_const', const=True, default=False, help='Test NN   (default: False)')


args = parser.parse_args()



def get_latest_file(directory, DNNorRetrain=''):
    list_of_files = glob.glob(directory+'/'+DNNorRetrain+'_*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# Define custom activation functions
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return 2*(x * torch.sigmoid(x))

class TanhPlusOne(nn.Module):
    def __init__(self):
        super(TanhPlusOne, self).__init__()

    def forward(self, x):
        return 2*(torch.tanh(x) + 1)



# Define LGK loss function
def lgk_loss_function(y_true, y_pred):
    alpha = torch.tensor(0.05)
    bandwith = torch.tensor(0.1)
    pi = torch.tensor(math.pi)
    norm = -1 / (bandwith * torch.sqrt(2 * pi))
    gaussian_kernel = norm * torch.exp(-torch.pow(y_pred / y_true - 1, 2) / (2 * (bandwith ** 2)))
    leakiness = alpha * torch.abs(y_pred / y_true - 1)
    lgk_loss = gaussian_kernel + leakiness
    loss = lgk_loss
    return loss.mean()

def lgk_loss_function_1(y_true, y_pred):
    alpha = torch.tensor(0.05)
    bandwith = torch.tensor(0.1)
    pi = torch.tensor(math.pi)
    norm = -1 / (bandwith * torch.sqrt(2 * pi))
    gaussian_kernel = norm * torch.exp(-torch.pow(y_pred / y_true - 1, 2) / (2 * (bandwith ** 2)))
    lgk_loss = gaussian_kernel
    loss = lgk_loss
    return loss.mean()


# Define the model architecture
class PUMitigation(nn.Module):
    def __init__(self, inputsize):
        super(PUMitigation, self).__init__()
        self.flatten = nn.Flatten()
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=inputsize, out_features=256)
        self.swish1 = Swish()
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.swish2 = Swish()
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.swish3 = Swish()
        self.fc4 = nn.Linear(in_features=64, out_features=8)
        self.swish4 = Swish()
        self.fc5 = nn.Linear(in_features=8, out_features=1)
        self.tanhPlusOne = TanhPlusOne()

    def forward(self, x):
        x = self.flatten(x)
        # x = self.dropout(x)
        x = self.swish1(self.fc1(x))
        x = self.swish2(self.fc2(x))
        x = self.swish3(self.fc3(x))
        x = self.swish4(self.fc4(x))
        x = self.tanhPlusOne(self.fc5(x))
        return x



def train_loop(dataloader, model, optimizer):

    model.train()
    loss_tot = 0
    n_batches = 0
    for batch, data in enumerate(dataloader):

        # Compute prediction and loss
        x = data[:, :-1].float()
        y = data[:, -1]

        # if batch==0: print(x)

        optimizer.zero_grad()
        out = model(x)

        loss = lgk_loss_function(y, out)

        loss.backward()
        optimizer.step()

        loss_tot += loss.item()
        n_batches +=1

    loss_tot /= n_batches
    return loss_tot

def val_loop(dataloader, model):

    loss_tot = 0
    n_batches = 0

    for batch, data in enumerate(dataloader):
        with torch.no_grad():
            # Compute prediction and loss
            x = data[:, :-1].float()
            y = data[:, -1]
            out = model(x)
            loss = lgk_loss_function(y, out)
            loss_tot += loss.item()
            n_batches +=1
    loss_tot /= n_batches

    return loss_tot

def retrain_loop(dataloader, model, optimizer):

    model.train()
    loss_tot = 0
    n_batches = 0
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        x = data[:, :-1].float()
        y = data[:, -1]
        optimizer.zero_grad()
        out = model(x)

        loss = lgk_loss_function_1(y, out)

        loss.backward()
        optimizer.step()

        loss_tot += loss.item()
        n_batches +=1

    loss_tot /= n_batches
    return loss_tot

def reval_loop(dataloader, model):

    loss_tot = 0
    n_batches = 0

    for batch, data in enumerate(dataloader):
        with torch.no_grad():
            # Compute prediction and loss
            x = data[:, :-1].float()
            y = data[:, -1]
            out = model(x)
            loss = lgk_loss_function_1(y, out)
            loss_tot += loss.item()
            n_batches +=1
    loss_tot /= n_batches

    return loss_tot

def main():

    dir_path = "ckpts/"
    try:
        os.system("mkdir {}".format(dir_path))
    except ImportError:
        print("{} already exists".format(dir_path))
    pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train dataset
    dataset_train = np.load('data/all_info_df_train.npy')
    x_train = dataset_train[:, 3:]
    y_train = dataset_train[:, 0]
    data_train = np.concatenate([x_train, y_train[:, None]], axis=-1)
    data_train = torch.from_numpy(data_train).to(device)
    print(f"Training dataset size {y_train.shape[0]}")

    # val dataset
    dataset_val = np.load('data/all_info_df_val.npy')
    x_val = dataset_val[:, 3:]
    y_val = dataset_val[:, 0]
    data_val = np.concatenate([x_val, y_val[:, None]], axis=-1)
    data_val = torch.from_numpy(data_val).to(device)
    print(f"Validation dataset size {y_val.shape[0]}")

    # test dataset
    dataset_test = np.load('data/all_info_df_test.npy')
    x_test = dataset_test[:, 3:]
    y_test = dataset_test[:, 0]
    data_test = np.concatenate([x_test, y_test[:, None]], axis=-1)
    data_test = torch.from_numpy(data_test).to(device)
    print(f"Test dataset size {y_test.shape[0]}")


    ## Make the input PyTorch-y
    train_loader = DataLoader(data_train, batch_size=4096, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=4096, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=4096, shuffle=False)

    num_features = x_train.shape[1]
    learning_rate = 0.0001


    if args.train:

        dnn_model = PUMitigation(num_features)
        optimizer = optim.Adam(dnn_model.parameters(), lr=learning_rate)
        dnn_model.to(device)
        nepochs = 5
        n_batches = 0

        # from torchsummary import summary
        # print(summary(dnn_model, (num_features,)))

        loss_train = []
        loss_val = []
        for epoch in range(nepochs):
            train_loss = train_loop(train_loader, dnn_model, optimizer)
            val_loss   = val_loop(val_loader, dnn_model)
            loss_train.append(train_loss)
            loss_val.append(val_loss)
            print(f'Epoch [{epoch+1}/{nepochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            name_ckpt = dir_path+"DNN_e{:03d}".format(epoch+1)+"_trainLoss{:.5f}".format(train_loss)+"_valLoss{:.5f}".format(val_loss)+".pt"

            torch.save({'epoch': epoch+1,
                        'model_state_dict': dnn_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss, 'val_Loss': val_loss, }, name_ckpt)

        fig, ax = plt.subplots()
        ax.plot(loss_train, label='loss')
        ax.plot(loss_val, label='val_loss')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + '/Losses_train_leakiness.png')
        plt.clf()

    if args.retrain:
        dnn_model = PUMitigation(num_features)
        optimizer = optim.Adam(dnn_model.parameters(), lr=learning_rate)
        dnn_model.to(device)
        # Load checkpoint
        ckpt_to_use = get_latest_file(dir_path, DNNorRetrain='DNN')
        checkpoint = torch.load(ckpt_to_use)
        dnn_model.load_state_dict(checkpoint['model_state_dict']) ## load model
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) ## load optimizer
        dnn_model.to(device)
        ## nepochs to retrain
        nepochs = 5

        loss_retrain = []
        loss_reval = []
        for epoch in range(nepochs):
            retrain_loss = retrain_loop(train_loader, dnn_model, optimizer)
            reval_loss   = reval_loop(val_loader, dnn_model)
            loss_retrain.append(retrain_loss)
            loss_reval.append(reval_loss)
            print(f'Epoch [{epoch+1}/{nepochs}], Train Loss: {retrain_loss:.4f}, Val Loss: {reval_loss:.4f}')
            name_ckpt = dir_path+"Retrain_DNN_e{:03d}".format(epoch+1)+"_trainLoss{:.5f}".format(retrain_loss)+"_valLoss{:.5f}".format(reval_loss)+".pt"

            torch.save({'epoch': epoch+1,
                        'model_state_dict': dnn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': retrain_loss,
                        'val_Loss': reval_loss,
                        }, name_ckpt)

        fig, ax = plt.subplots()
        ax.plot(loss_train, label='loss')
        ax.plot(loss_val, label='val_loss')
        ax.set_xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir_path + '/Losses_final.png')
        plt.clf()
        pass


    if args.test:
        dnn_model = CustomModel(num_features)
        dnn_model.to(device)


        ckpt_to_use = get_latest_file(dir_path, DNNorRetrain='Retrain')
        checkpoint = torch.load(ckpt_to_use)
        dnn_model.load_state_dict(checkpoint['model_state_dict']) ## load model

        dnn_model.eval()

        predictions, y, x_tests = [], [], []
        for batch, data in enumerate(test_loader):
            x_test = data[:, :-1].float()
            y_test = data[:, -1]
            out = dnn_model(x_test).detach().cpu().numpy()
            predictions.append(out)
            y.append(y_test.cpu().detach().numpy())
            x_tests.append(x_test.cpu().detach().numpy())


        y = np.concatenate(y)
        predictions = np.concatenate(predictions)
        out = np.concatenate(predictions, axis=0)
        # output = np.concatenate(out1, axis=0)

        x_test = np.concatenate(x_tests)

        np.save(dir_path + '/trueResponse.npy', y)
        np.save(dir_path + '/predResponse.npy', out)
        np.save(dir_path + '/x_test.npy', x_test)

        pass


    return



# Main function call.
if __name__ == '__main__':
    main()
    pass
