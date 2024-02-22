import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import TensorDataset, DataLoader
import argparse
import glob

parser = argparse.ArgumentParser(description='Perform signal injection test.')
parser.add_argument('--train',    dest='train',    action='store_const', const=True, default=False, help='Train NN  (default: False)')
parser.add_argument('--retrain',  dest='retrain',  action='store_const', const=True, default=False, help='Retrain NN  (default: False)')
parser.add_argument('--test',     dest='test',     action='store_const', const=True, default=False, help='Test NN   (default: False)')
parser.add_argument('--closure',  dest='closure',  action='store_const', const=True, default=False, help='Closure')
parser.add_argument('--weight',   dest='weight',   type=int, default=0, help='Weight: 0: no weight 1: response, 2: response wider, 3: energy, 4: log energy')
parser.add_argument('--outdir',   dest='outdir',   type=str, default='', help='Directory with output is stored')


args = parser.parse_args()

def get_latest_file(directory, DNNorRetrain=''):
    list_of_files = glob.glob(directory+'/'+DNNorRetrain+'_*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
    
# Define custom activation functions
class Swish(nn.Module):
    def forward(self, x):
        return 2*(x * torch.sigmoid(x))

class TanhPlusOne(nn.Module):
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
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16, out_features=256)
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
        x = self.swish1(self.fc1(x))
        x = self.swish2(self.fc2(x))
        x = self.swish3(self.fc3(x))
        x = self.swish4(self.fc4(x))
        x = self.tanhPlusOne(self.fc5(x))
        return x

# Build and compile the model
def build_and_compile_model(X_train, lr):
    model = CustomModel()
    criterion = lgk_loss_function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def build_and_compile_model_retrain(X_train, lr):
    model = CustomModel()
    criterion = lgk_loss_function_1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def main():

    dir_path = "ckpts/"
    try:
        os.system("mkdir {}".format(dir_path))
    except ImportError:
        print("{} already exists".format(dir_path))
    pass

    # train dataset
    dataset_train = np.load('data/all_info_df_train.npy')
    x_train = dataset_train[:, 4:]
    y_train = dataset_train[:, 0]
    # w_train = dataset_train[:, 4] ## 1: response, 2: response wider, 3: energy, 4: log energy
    data_train = np.concatenate([x_train, y_train[:, None]], axis=-1)
    print(f"Training dataset size {y_train.shape[0]}")

    # val dataset
    dataset_val = np.load('data/all_info_df_val.npy')
    x_val = dataset_val[:, 4:]
    y_val = dataset_val[:, 0]
    data_val = np.concatenate([x_val, y_val[:, None]], axis=-1)
    print(f"Validation dataset size {y_val.shape[0]}")

    # test dataset
    dataset_test = np.load('data/all_info_df_test.npy')
    x_test = dataset_test[:, 4:]
    y_test = dataset_test[:, 0]
    data_test = np.concatenate([x_test, y_test[:, None]], axis=-1)
    print(f"Test dataset size {y_test.shape[0]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # criterion.to(device)

    ## Make the input PyTorch-y
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    dataset_train = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset_train, batch_size=4096, shuffle=True)

    dataset_val = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset_val, batch_size=4096, shuffle=False)

    dataset_test = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset_test, batch_size=4096, shuffle=False)


    if args.train:

        # Build and compile the model
        dnn_model, criterion, optimizer = build_and_compile_model(x_train, lr=0.0001)
        dnn_model.to(device)
        nepochs = 5
    
        for epoch in range(nepochs):
            dnn_model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = dnn_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)

            # Evaluate on validation set
            dnn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = dnn_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(val_loader.dataset)

            print(f'Epoch [{epoch+1}/{nepochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            name_ckpt = dir_path+"DNN_e{:03d}".format(epoch+1)+"_trainLoss{:.5f}".format(train_loss)+"_valLoss{:.5f}".format(val_loss)+".pt"

            torch.save({'epoch': epoch+1,
                        'model_state_dict': dnn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_Loss': val_loss,
                        }, name_ckpt)
            
    if args.retrain:
        dnn_model, criterion, optimizer = build_and_compile_model_retrain(x_train, lr=0.0001)
        dnn_model.to(device)
        # Load checkpoint
        ckpt_to_use = get_latest_file(dir_path, DNNorRetrain='DNN')
        checkpoint = torch.load(ckpt_to_use)
        dnn_model.load_state_dict(checkpoint['model_state_dict']) ## load model
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) ## load optimizer 
        dnn_model.to(device)
        ## nepochs to retrain
        nepochs = 5

        for epoch in range(nepochs):
            dnn_model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = dnn_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)

            # Evaluate on validation set
            dnn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = dnn_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(val_loader.dataset)

            print(f'Epoch [{epoch+1}/{nepochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            name_ckpt = dir_path+"Retrain_DNN_e{:03d}".format(epoch+1)+"_trainLoss{:.5f}".format(train_loss)+"_valLoss{:.5f}".format(val_loss)+".pt"
            torch.save({'epoch': epoch+1,
                        'model_state_dict': dnn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_Loss': val_loss,
                        }, name_ckpt)
        pass

    if args.test:

        dnn_model, criterion, _ = build_and_compile_model_retrain(x_train, lr=0.0001)
        dnn_model.to(device)


        ckpt_to_use = get_latest_file(dir_path, DNNorRetrain='Retrain')
        checkpoint = torch.load(ckpt_to_use)
        dnn_model.load_state_dict(checkpoint['model_state_dict']) ## load model

        dnn_model.eval()

        out = [] 
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out.append(dnn_model(batch_x).detach().cpu().numpy())


        out = np.array(out)
        out1 = np.concatenate(out, axis=0)
        output = np.concatenate(out1, axis=0)
        
        np.save(dir_path + '/trueResponse.npy', y_test)
        np.save(dir_path + '/predResponse.npy', output)
        np.save(dir_path + '/x_test.npy', x_test)

        # all_tested = np.column_stack((x_test, y_test)) 
        # all_tested = np.column_stack((all_tested, out)) 
        # np.save(dir_path + '/all_tested.npy', all_tested)
        pass


    ## test
    return





# Main function call.
if __name__ == '__main__':
    main()
    pass
