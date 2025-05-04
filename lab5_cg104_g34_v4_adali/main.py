# main.py
import os, random, time, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST

from helpers import calculate_accuracy,EarlyStopping
# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--hidden_layers", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--loss_fn", type=str, choices=["mse", "mae", "crossentropy"], required=True)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

# Seed
SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Data
dataset = FashionMNIST(root='./data', train=True, download=True)
testset = FashionMNIST(root='./data', train=False, download=True)

X = dataset.data.float().view(-1, 784) / 255.0
y = dataset.targets
X_train,X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
y_train,y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
X_test = testset.data.float().view(-1, 784) / 255.0
y_test = testset.targets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)
X_test,y_test = X_test.to(device), y_test.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)


# Model
class FashionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, width, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = FashionClassifier(784, args.hidden_layers, args.width, 10).to(device)

# Loss Function
if args.loss_fn == "mse":
    loss_fn = nn.MSELoss()
elif args.loss_fn == "mae":
    loss_fn = nn.L1Loss()
else:
    loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training Loop
n_epochs = args.epochs
losses = pd.DataFrame(columns=["Learning Step", "Loss"])
train_df = pd.DataFrame(columns=["Epoch", "Loss", "Train Accuracy", "Test Accuracy", "Time"])
early_stopper = EarlyStopping(tolerance=5, min_delta=0.05)
min_delta_loss = 0.0003
consecutive_no_improvement = 0
patience = 3
best_epoch_loss = float("inf")
for epoch in range(n_epochs):
    model.train()
    epoch_start = time.time()
    for i in range(0, len(X_train), args.batch_size):
        Xbatch = X_train[i:i+args.batch_size]
        ybatch = y_train[i:i+args.batch_size]
        optimizer.zero_grad()
        output = model(Xbatch)
        target = ybatch if args.loss_fn == "crossentropy" else nn.functional.one_hot(ybatch, num_classes=10).float()
        loss = loss_fn(output, target)
        losses.loc[len(losses)] = [i, loss.item()]
        loss.backward()
        optimizer.step()
    
    

    epoch_time = time.time() - epoch_start
   
        

    model.eval()
    with torch.no_grad():
        pred_train = model(X_train)
        train_preds = pred_train.argmax(dim=1)
        acc = calculate_accuracy(y_train, train_preds)
        val_preds = model(X_val).argmax(dim=1)
        val_acc = calculate_accuracy(y_val, val_preds)
    
    

    train_df.loc[epoch] = [epoch+1, loss.item(), acc, val_acc, epoch_time]
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f}, Train Accuracy: {acc:.4f}, Validation Accuracy: {val_acc:.4f}, Time: {epoch_time:.2f}s")
    early_stopper(acc, val_acc)
    if early_stopper.early_stop:
      print("We are stopping early at epoch:", i)
      break
    current_epoch_loss = loss.item()
    if best_epoch_loss - current_epoch_loss >= min_delta_loss:
        best_epoch_loss = current_epoch_loss
        consecutive_no_improve = 0
    else:
        consecutive_no_improve += 1
        if consecutive_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} as the loss did not improve by at least {min_delta_loss} for {patience} consecutive epochs.")
            break


# Save results
params = f"_{SEED}_{n_epochs}_{args.batch_size}_{args.lr}_{args.loss_fn}_{args.hidden_layers}hl_{args.width}w"
train_results_path = f"training_results/result_{params}"
os.makedirs(train_results_path, exist_ok=True)
train_df.to_csv(f"{train_results_path}/train_df.csv", index=False)
print(f"Prediction results on test set: {calculate_accuracy(y_test, model(X_test).argmax(dim=1)):.4f}")
# Plot loss vs learning step
plt.figure(figsize=(12, 5))
plt.plot(losses["Loss"], label="Loss")
plt.xlabel("Learning Step")
plt.ylabel("Loss")
plt.title("Loss vs Learning Step")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{train_results_path}/loss_vs_learning_step.png")

# Plot loss vs epoch
plt.figure(figsize=(12, 5))
plt.plot(train_df["Epoch"], train_df["Loss"], label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{train_results_path}/loss_vs_epoch.png")

# Plot accuracy on training and test sets vs epoch
plt.figure(figsize=(12, 5))
plt.plot(train_df["Epoch"], train_df["Train Accuracy"], label="Train Accuracy")
plt.plot(train_df["Epoch"], train_df["Test Accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{train_results_path}/accuracy_vs_epoch.png")
