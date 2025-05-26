import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
import time
import psutil
import csv

from utils import *

#TODO: Implement the parallelization for ViT model
def parallelize_vit_model(model, mesh):
    pass

def train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                num_epochs:int, optimizer:torch.optim, train_loss:torch.nn, val_loss:callable, device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensor Parallelism Example")
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k",
                        help="Name of the ViT model to use")
    parser.add_argument("--minibatch", type=int, default=512, 
                        help="Size of the minibatch for training")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for the optimizer")
    parser.add_argument("--tau", type=int, default=10,
                        help="Number of iteration to be performed before synchronizing")
    args = parser.parse_args()

    rank, world, device = init_distributed()

    model_name = args.model
    minibatch = args.minibatch
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    tau = args.tau

    image_processor, model = load_model(model_name, num_labels=101)

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = image_processor(images=list(images), return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device_name, (world,)) #1D Parallelism

    model = parallelize_module(model, mesh)

    # train_set, val_set, _ = load_dataset("../data")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch, shuffle=True, collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=minibatch, shuffle=False, collate_fn=collate_fn)

    # optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # criterion = torch.nn.CrossEntropyLoss()

    # model_name = model_name.split("/")[-1] 
    # train(model, train_loader, val_loader, num_epochs, optim, criterion, accuracy, device=device, model_name=model_name)
    

