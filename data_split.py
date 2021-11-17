from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch 

# Train and validation split function
def split_training_dataset(dataset, batch_size_train, val_split=(1/6)):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    train_loader = torch.utils.data.DataLoader( datasets['train'], batch_size=batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader( datasets['val'], batch_size=batch_size_train, shuffle=True)
    return train_loader, val_loader