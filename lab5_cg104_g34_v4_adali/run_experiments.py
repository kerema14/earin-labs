# run_experiments.py
import os

experiments = [
   
    {"epochs": 300, "lr": 0.0000205, "batch_size": 512, "hidden_layers": 2, "width": 4192, "loss_fn": "crossentropy"},
    """
    {"lr": 0.0000235, "batch_size": 512, "hidden_layers": 1, "width": 128, "loss_fn": "mae"},
    {"lr": 0.0000235, "batch_size": 512, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},  # mainly for testing
    {"lr": 0.00005, "batch_size": 16, "hidden_layers": 0, "width": 0, "loss_fn": "croessentropy"},
    {"lr": 0.0000235, "batch_size": 16, "hidden_layers": 2, "width": 256, "loss_fn": "mae"},
    {"lr": 0.0001, "batch_size": 8, "hidden_layers": 1, "width": 512, "loss_fn": "crossentropy"},
    {"lr": 0.00005, "batch_size": 1, "hidden_layers": 0, "width": 0, "loss_fn": "mse"},
    """
    # Additional parameter sets
    
]

for i, exp in enumerate(experiments):
    args = " ".join(f"--{k} {v}" for k, v in exp.items())
    print(f"\nRunning experiment {i+1} with args: {args}")
    os.system(f"python main.py {args}")
