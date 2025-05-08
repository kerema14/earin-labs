# run_experiments.py
import os

experiments = [
    #lr: 5e-5, 2.05e-5, 2.35e-5, 2.5e-5, 3.0e-5,1.0e-4, 2.35e-4

    
    
    {"epochs": 305, "lr": 0.0000205, "batch_size": 1, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},
    {"epochs": 305, "lr": 0.0000205, "batch_size": 8, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},
    {"epochs": 305, "lr": 0.0000205, "batch_size": 16, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},
    {"epochs": 305, "lr": 0.0000205, "batch_size": 64, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},
    {"epochs": 305, "lr": 0.0000205, "batch_size": 256, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},
    {"epochs": 305, "lr": 0.0000205, "batch_size": 512, "hidden_layers": 1, "width": 128, "loss_fn": "crossentropy"},

    
    
    # Additional parameter sets
    
]

for i, exp in enumerate(experiments):
    args = " ".join(f"--{k} {v}" for k, v in exp.items())
    print(f"\nRunning experiment {i+1} with args: {args}")
    os.system(f"python main.py {args}")
