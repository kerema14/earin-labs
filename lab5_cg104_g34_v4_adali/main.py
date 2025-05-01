import os
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score

import torch.nn as nn
import torch.optim as optim

dataset = FashionMNIST(root='./data', train=True, download=True)
testset = FashionMNIST(root='./data', train=False, download=True)

SEED  = 69
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class fashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 512)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(256, 10)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x
    

model = fashionClassifier().to('cuda') # Move model to GPU
loss_fn = nn.CrossEntropyLoss().to('cuda') # Move loss function to GPU
optimizer = optim.Adam(model.parameters(), lr=0.0000235,weight_decay=0.0001) # Adam optimizer with weight decay
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5) # Learning rate scheduler
 # Learning rate scheduler 

Xcpu = dataset.data
ycpu = dataset.targets
Xcpu = Xcpu.float() / 255.0 # Normalize the data to [0, 1]
Xcpu = Xcpu.view(Xcpu.shape[0], -1) # Flatten the images to 784-dimensional vectors
X = Xcpu.to('cuda')
y = ycpu.to('cuda') # Move labels to GPU
X_test  = testset.data
X_test = X_test.float() / 255.0 # Normalize the test data to [0, 1]
X_test = X_test.view(X_test.shape[0], -1) # Flatten the test images to 784-dimensional vectors
y_test = testset.targets 
set_seed(SEED)


n_epochs = 100
batch_size = 10

losses = []
train_df = pd.DataFrame(columns=["Epoch", "Loss", "Train Accuracy", "Train Precision", "Test Accuracy", "Test Precision"])

for epoch in range(n_epochs):
    start_time = time.time()
    print(f"Epoch {epoch+1}/{n_epochs}")
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y[i:i+batch_size]
        y_pred = model(Xbatch).to('cuda') # Move predictions to GPU
        loss = loss_fn(y_pred, ybatch)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - start_time
    
    # Evaluate metrics on the full training set
    with torch.no_grad():
        outputs = model(X).cpu() # Move outputs to CPU for evaluation
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy_score(ycpu.numpy(), predictions)
        pr = precision_score(ycpu.numpy(), predictions, average='macro')
        test_acc = accuracy_score(y_test.numpy(), torch.argmax(model(X_test.to('cuda')).cpu(), dim=1))
        test_pr = precision_score(y_test.numpy(), torch.argmax(model(X_test.to('cuda')).cpu(), dim=1), average='macro')
    
    
    print(f'Epoch {epoch+1}: Loss {loss:.4f},Train Accuracy {acc:.4f},Train Precision {pr:.4f}, Time: {epoch_time:.2f} sec')
    print(f'Test Accuracy {test_acc:.4f}, Test Precision {test_pr:.4f}')
    train_df.loc[epoch] = [epoch+1, loss.item(), acc, pr, test_acc, test_pr] #
    
#check if there is already a train_df.csv file and if so, create a new one with a different name
params = f"_{SEED}_{n_epochs}_{batch_size}_{optimizer.defaults['lr']}_{loss_fn.__class__.__name__}_{model.__class__.__name__}"
train_results_path = f"training_results/{params}/"
if os.path.exists(f"train_df.csv"):
    i = 1
    while os.path.exists(f"train_df-{i}.csv"):
        i += 1
    
    train_df.to_csv(f"train_df-{i}.csv", index=False)
else:
    #save the dataframe to a csv file
    
    train_df.to_csv(f"train_df.csv", index=False)
"""
please create plots visualizing:
• The loss value for every learning step,
• Accuracy on the training and validation set after each epoch.
"""
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


# Plot loss vs learning step
plt.figure(figsize=(12, 5))
plt.plot(losses, label="Loss")
plt.xlabel("Learning Step")
plt.ylabel("Loss")
plt.title("Loss vs Learning Step")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{train_results_path}/loss_vs_learning_step.png")


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



