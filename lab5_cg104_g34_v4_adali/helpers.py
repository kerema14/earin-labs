import torch
def calculate_accuracy(labels:torch.Tensor, predictions:torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions.
    
    Args:
        labels (torch.Tensor): True labels.
        predictions (torch.Tensor): Predicted labels.
        
    Returns:
        float: Accuracy score.
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True