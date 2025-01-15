# %%
# load packages
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from torchinfo import summary
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
class TABL_layer(nn.Module):
    def __init__(self, d2, d1, t1, t2):
        super().__init__()
        self.t1 = t1

        weight = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')
        
        weight2 = torch.Tensor(t1, t1)
        self.W = nn.Parameter(weight2)
        nn.init.constant_(self.W, 1/t1)
 
        weight3 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias1 = torch.Tensor(d2, t2)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        l = torch.Tensor(1,)
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)

        self.activation = nn.ReLU()

    def forward(self, X):
        
        #maintaining the weight parameter between 0 and 1.
        if (self.l[0] < 0): 
          l = torch.Tensor(1,).to(device)
          self.l = nn.Parameter(l)
          nn.init.constant_(self.l, 0.0)

        if (self.l[0] > 1): 
          l = torch.Tensor(1,).to(device)
          self.l = nn.Parameter(l)
          nn.init.constant_(self.l, 1.0)
     
        #modelling the dependence along the first mode of X while keeping the temporal order intact (7)
        X = self.W1 @ X

        #enforcing constant (1) on the diagonal
        W = self.W -self.W *torch.eye(self.t1,dtype=torch.float32).to(device)+torch.eye(self.t1,dtype=torch.float32).to(device)/self.t1

        #attention, the aim of the second step is to learn how important the temporal instances are to each other (8)
        E = X @ W

        #computing the attention mask  (9)
        A = torch.softmax(E, dim=-1)

        #applying a soft attention mechanism  (10)
        #he attention mask A obtained from the third step is used to zero out the effect of unimportant elements
        X = self.l[0] * (X) + (1.0 - self.l[0])*X*A

        #the final step of the proposed layer estimates the temporal mapping W2, after the bias shift (11)
        y = X @ self.W2 + self.B
        return y

class BL_layer(nn.Module):
  def __init__(self, d2, d1, t1, t2):
        super().__init__()
        weight1 = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight1)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')

        weight2 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight2)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias1 = torch.zeros((d2, t2))
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        self.activation = nn.ReLU()

  def forward(self, x):

    x = self.activation(self.W1 @ x @ self.W2 + self.B)

    return x

# %%

class BiN(nn.Module):
    def __init__(self, d1, t1):
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        # self.t2 = t2
        # self.d2 = d2

        bias1 = torch.Tensor(t1, 1)
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)

        l1 = torch.Tensor(t1, 1)
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)

        bias2 = torch.Tensor(d1, 1)
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        l2 = torch.Tensor(d1, 1)
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)

        y1 = torch.Tensor(1, )
        self.y1 = nn.Parameter(y1)
        nn.init.constant_(self.y1, 0.5)

        y2 = torch.Tensor(1, )
        self.y2 = nn.Parameter(y2)
        nn.init.constant_(self.y2, 0.5)

    def forward(self, x):

        # if the two scalars are negative then we setting them to 0
        if (self.y1[0] < 0):
            y1 = torch.FloatTensor(1).to(x.device)
            self.y1 = nn.Parameter(y1)
            nn.init.constant_(self.y1, 0.01)

        if (self.y2[0] < 0):
            y2 = torch.FloatTensor(1).to(x.device)
            self.y2 = nn.Parameter(y2)
            nn.init.constant_(self.y2, 0.01)

        # normalization along the temporal dimensione
        T2 = torch.ones([self.t1, 1], device=device)
        x2 = torch.mean(x, dim=2)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))
        
        std = torch.std(x, dim=2)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))
        # it can be possible that the std of some temporal slices is 0, and this produces inf values, so we have to set them to one
        std[std < 1e-4] = 1

        diff = x - (x2 @ (T2.T))
        Z2 = diff / (std @ (T2.T))

        X2 = self.l2 @ T2.T
        X2 = X2 * Z2
        X2 = X2 + (self.B2 @ T2.T)

        # normalization along the feature dimension
        T1 = torch.ones([self.d1, 1], device=device)
        x1 = torch.mean(x, dim=1)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))

        std = torch.std(x, dim=1)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))

        op1 = x1 @ T1.T
        op1 = torch.permute(op1, (0, 2, 1))

        op2 = std @ T1.T
        op2 = torch.permute(op2, (0, 2, 1))

        z1 = (x - op1) / (op2)
        X1 = (T1 @ self.l1.T)
        X1 = X1 * z1
        X1 = X1 + (T1 @ self.B1.T)

        # weighing the imporance of temporal and feature normalization
        x = self.y1 * X1 + self.y2 * X2

        return x 

# %%
class BiN_BTABL(nn.Module):
  def __init__(self, d2, d1, t1, t2, d3, t3):
    super().__init__()

    self.BiN = BiN(d1, t1)
    self.BL = BL_layer(d2, d1, t1, t2)
    self.TABL = TABL_layer(d3, d2, t2, t3)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    #first of all we pass the input to the BiN layer, then we use the B(TABL) architecture
    x = self.BiN(x)

    self.max_norm_(self.BL.W1.data)
    self.max_norm_(self.BL.W2.data)
    x = self.BL(x)
    x = self.dropout(x)

    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    x = torch.squeeze(x)
    x = torch.softmax(x, 1)


    return x

  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))


class BiN_CTABL(nn.Module):
  def __init__(self, d1, t1, d2, t2, d3, t3, d4, t4):
    super().__init__()

    self.BiN = BiN(d1, t1)
    self.BL = BL_layer(d2, d1, t1, t2)
    self.BL2 = BL_layer(d3, d2, t2, t3)
    self.TABL = TABL_layer(d4, d3, t3, t4)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    #first of all we pass the input to the BiN layer, then we use the C(TABL) architecture
    x = self.BiN(x)

    self.max_norm_(self.BL.W1.data)
    self.max_norm_(self.BL.W2.data)
    x = self.BL(x)
    x = self.dropout(x)
    
    self.max_norm_(self.BL2.W1.data)
    self.max_norm_(self.BL2.W2.data)
    x = self.BL2(x)
    x = self.dropout(x)

    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    x = torch.squeeze(x, dim=-1)
    x = torch.softmax(x, 1)
    
    return x

  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm)) 

# %%
class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y, num_classes, time_window):
        """Initialization""" 
        self.num_classes = num_classes
        self.time_window = time_window
        self.x = x   
        self.y = y

        self.length = x.shape[0] -self.time_window + 1

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return int(self.length)

    def __getitem__(self, i):
        input = self.x[i:i+self.time_window, :]
        input = input.permute(1, 2, 0)
        input = torch.squeeze(input)

        return input, self.y[i]
  
def batch_gd(model, train_loader, val_loader, criterion, optimizer, epochs, model_save_name="best", waiting_pacience=10):
    
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    test_loss_goup_cnt = 0
    
    
       
    for it in range(epochs):
        if optimizer.param_groups[0]["lr"] < 1e-5:
          print("Early stopping at", it)
          break

        if test_loss_goup_cnt > waiting_pacience:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
            test_loss_goup_cnt = 0
            print("LR decayed at Epoch.", it, "to", g['lr'])

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in tqdm(train_loader):
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            #computing the error
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            
        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        model.eval()
        
        test_loss = []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)      
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        #We save the best model
        if test_loss < best_test_loss:
            torch.save(model, f'./{model_save_name}.pt')
            best_test_loss = test_loss
            test_loss_goup_cnt = 0
            best_test_epoch = it
            print('model saved')
        else:
           test_loss_goup_cnt += 1

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')
        
    #torch.save(model, '/content/drive/MyDrive/Output/best_model_translob_FI')
    return train_losses, test_losses

def objective(config):
  time_now = datetime.now().strftime("%Y%m%d%H%M%S")
  print(f"Sweep at {time_now}, Params:", config)

  # config: dataset_norm, horizon, label_percentage, time_window, lob_depth, d2, t2, d3, t3
  symbol = config.symbol
  epochs = config.epochs
  dataset_norm = config.dataset_norm
  horizon = config.horizon
  label_percentage = config.label_percentage
  time_window = config.time_window
  lob_depth = config.lob_depth
  d2 = lob_depth*8
  d3 = lob_depth*8
  t2 = config.time_window // 2
  t3 = config.time_window // 4
  # d2 = config.d2
  # t2 = config.t2
  # d3 = config.d3
  # t3 = config.t3

  # %%
  # Load Data
  import sys
  sys.path.append("..")
  from utils.tool import load_fi_2010, load_my

  if symbol == "FI2010":
    if dataset_norm == "znorm":
      dataset_norm = "ZScore"
    elif dataset_norm == "decpre":
      dataset_norm = "DecPre"
    data_train, data_val, data_test = load_fi_2010(dataset_norm)
  else:
    if dataset_norm == "znorm":
      dataset_norm = "zscore"
    elif dataset_norm == "decpre":
      dataset_norm = "dec"
    data_train, data_val, data_test = load_my(symbol, dataset_norm, dataset_path="../data_my2")

  if symbol == "FI2010":
    horizons = {
      1:-5,
      2:-4,
      3:-3,
      5:-2,
      10:-1
    }
  else:
    horizons = {
      1:-10,
      2:-9,
      3:-8,
      5:-7,
      10:-6,
      50:-5,
      100:-4,
      200:-3,
      500:-2,
      1000:-1
    }

  y_pct_train = data_train[horizons[horizon], :].flatten()
  y_pct_val = data_val[horizons[horizon], :].flatten()
  y_pct_test = data_test[horizons[horizon], :].flatten()

  thres = np.percentile(np.abs(y_pct_train), label_percentage)

  y_train = np.zeros_like(y_pct_train)
  y_train[y_pct_train > thres] = 1
  y_train[y_pct_train < -thres] = -1

  y_val = np.zeros_like(y_pct_val)
  y_val[y_pct_val > thres] = 1
  y_val[y_pct_val < -thres] = -1

  # y_test = np.zeros_like(y_pct_test)
  # y_test[y_pct_test > thres] = 1
  # y_test[y_pct_test < -thres] = -1
  y_test = y_pct_test

  y_train = y_train[time_window-1:] + 1
  y_val = y_val[time_window-1:] + 1
  # y_test = y_test[time_window-1:] + 1 

  data_train = data_train[:4*lob_depth, :].T
  data_val = data_val[:4*lob_depth, :].T
  data_test = data_test[:4*lob_depth, :].T

  print("train:", data_train.shape)
  print("val: ", data_val.shape)
  print("test: ", data_test.shape)


  # %%
  #Computing weights for the weighted cross entropy loss
  def compute_weights(y):
    cont_0 = 0
    cont_1 = 0
    cont_2 = 0
    for i in range(y.shape[0]):
      if (y[i] == 0):
        cont_0 += 1
      elif (y[i] == 1):
        cont_1 += 1
      elif (y[i] == 2):
        cont_2 += 2
      else: 
        raise Exception("wrong labels")
    print(f"0:{cont_0}, 1:{cont_1}, 2:{cont_2}")
    return torch.Tensor([1e6/cont_0, 1e6/cont_1, 1e6/cont_2]).to(device)

  # y_total = np.concatenate((y_train, y_val, y_test))
  weights = compute_weights(y_train)

  # %%
  #Hyperparameters
  batch_size = 256 
  lr = 0.001
  num_classes = 3

  dataset_train = Dataset(data_train, y_train, num_classes, time_window)
  dataset_val = Dataset(data_val, y_val, num_classes, time_window)
  dataset_test = Dataset(data_test, y_test, num_classes, time_window)

  train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

  # %%
  #you can choose between the two architectures

  # model = BiN_BTABL(4*lob_depth, time_window, 120, 5, num_classes, 1)
  model = BiN_CTABL(4*lob_depth, time_window, d2, t2, d3, t3, num_classes, 1)

  # %%
  # summary(model, (1, 4*lob_depth, time_window))

  model.to(device)

  criterion = nn.CrossEntropyLoss(weight=weights)
  optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-3)

  train_losses, val_losses = batch_gd(model, train_loader, val_loader, criterion, optimizer, 
                                      epochs, model_save_name=time_now)

  model = torch.load(f'./{time_now}.pt')
  model.eval()

  n_correct = 0.
  n_total = 0.
  all_targets = []
  all_predictions = []
  total_pnl = 0
  total_win_times = 0
  total_trade_times = 0
  pnl_ratios = []

  for inputs, targets in tqdm(test_loader):
      # Move to GPU
      inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)

      # Forward pass
      with torch.no_grad():
        outputs = model(inputs)
      #outputs = torch.squeeze(outputs)
      # Get prediction
      # torch.max returns both max and argmax
      _, predictions = torch.max(outputs, 1)

      targets = targets.detach().cpu().numpy()
      predictions = predictions.detach().cpu().numpy()

      # update counts
      targets_labels = np.zeros_like(targets) + 1
      targets_labels[targets>thres] = 0
      targets_labels[targets<-thres] = 2
      n_correct += (predictions == targets_labels).sum().item()
      n_total += targets.shape[0]
      all_targets.append(targets_labels)
      all_predictions.append(predictions)

      # backtest
      long_pnl = targets[predictions == 0]
      short_pnl = -targets[predictions == 2]

      win_times = (long_pnl>0).sum() + (short_pnl>0).sum()
      trade_times = (predictions == 0).sum() + (predictions == 2).sum()

      long_pnl_sum = long_pnl.sum()
      short_pnl_sum = short_pnl.sum()
      transaction_fees = 0 # 1e-4 * trade_times

      total_pnl += (long_pnl_sum + short_pnl_sum - transaction_fees)
      total_win_times += win_times
      total_trade_times += trade_times
      pnl_ratios.append((long_pnl_sum+short_pnl_sum)/(trade_times+1e-7))

  test_acc = n_correct / n_total
  print(f"Test acc: {test_acc:.4f}")
  print(f"Total PNL: {total_pnl:.4f}")
  win_rates = total_win_times/total_trade_times
  pnl_ratio_per_trade = np.mean(pnl_ratios)
  print(f"Win Rates: {100*win_rates:.2f}%, Trade Times: {total_trade_times}")
  print(f"PNL Ratio Per Trade: {pnl_ratio_per_trade:.8f}")

  all_targets = np.concatenate(all_targets)    
  all_predictions = np.concatenate(all_predictions)  
  print(classification_report(all_targets, all_predictions, digits=4))

  return {
     "test_acc":test_acc,
     "total_pnl":total_pnl,
     "total_trade_times":total_trade_times,
     "win_rates":win_rates, 
     "pnl_ratio_per_trade":pnl_ratio_per_trade
     }

if __name__ == "__main__":
  configuration = {
         "symbol": "futures_ethusdt",
         "epochs": 100,
         "dataset_norm": "znorm",
         "horizon": 10,
         "label_percentage": 80,
         "time_window": 50,
         "lob_depth": 25,
        #  "d2": {'distribution': 'q_log_uniform_values', 'q': 8, "max": 128, "min": 16},
        #  "t2": {'distribution': 'q_log_uniform_values', 'q': 8, "max": 128, "min": 16},
        #  "d3": {'distribution': 'q_log_uniform_values', 'q': 8, "max": 128, "min": 16},
        #  "t3": {'distribution': 'q_log_uniform_values', 'q': 8, "max": 128, "min": 16},
      }
  
  from collections import namedtuple
  Configuration = namedtuple('Configuration', configuration.keys())
  configuration = Configuration(**configuration)

  objective(configuration)
