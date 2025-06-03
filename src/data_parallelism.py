import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import psutil
import csv
from utils import *

from transformers import logging, set_seed
logging.set_verbosity_error()

torch.set_default_device("cpu")
set_seed(seed=42, deterministic=False)

torch.set_num_threads(28)
torch.set_num_interop_threads(28)

def synch_train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                num_epochs:int, optimizer:torch.optim, train_loss:torch.nn, val_loss:callable, device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    

    model.to(device)

    # csv file to save the stats
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = train_loader.batch_size * world_size
    iface = os.environ.get('IFACE')
    file_name = f"../log/synch_ddp/rank_{rank}_synch_ddp_{world_size}_minibatch_{batch_size}_{model_name}_model_{iface}.csv"

    model = DDP(model)

    with open(file_name, mode="w+") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "batch_id", "loss", "forward_time", "backward_time", "phase"])

        for epoch in range(num_epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                images = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                start_forward = time.time()
                outputs = model(images)
                end_forward = time.time()

                loss = train_loss(outputs.logits, labels)

                start_backward = time.time()
                loss.backward()
                end_backward = time.time()
                
                optimizer.step()

                # Log the stats
                writer.writerow([epoch, i, loss.item(), end_forward-start_forward, end_backward-start_backward, "train"])
                file.flush()

            # Validation step
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    images = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    start_forward = time.time()
                    outputs = model(images)
                    end_forward = time.time()

                    accuracy = val_loss(outputs.logits, labels)
                    # Log the stats
                    writer.writerow([epoch, i, accuracy, end_forward - start_forward, 0, "val"])
            
            file.flush()

@torch.no_grad()
def average_weights(model: torch.nn.Module, world_size: int):
    """Average the model weights across all processes using coalesced communication."""
    # Flatten all parameters that require grad into a single contiguous tensor
    param_views = []
    for param in model.parameters():
        if param.requires_grad:
            param_views.append(param.data.view(-1))
    
    # Concatenate into a flat tensor
    flat_tensor = torch.cat(param_views)

    # Perform all-reduce averaging
    dist.all_reduce(flat_tensor, op=dist.ReduceOp.SUM)
    flat_tensor /= world_size

    # Unflatten back into model parameters
    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.data.copy_(flat_tensor[offset:offset + numel].view_as(param))
            offset += numel

def asynch_train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader,
                  num_epochs:int, tau:int, optimizer:torch.optim, criterion:torch.nn, val_loss:callable, device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    
    model.to(device)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = train_loader.batch_size * world_size
    iface = os.environ.get('IFACE')
    file_name = f"../log/asynch_ddp/rank_{rank}_asynch_ddp_{world_size}_minibatch_{batch_size}_tau_{tau}_{model_name}_model_{iface}.csv"
    model = DDP(model)

    iteration = 0

    with open(file_name, mode="w+") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "batch_id", "loss", "forward_time", "backward_time", "synch_avg_time", "phase"])

        for epoch in range(num_epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                images = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                start_forward = time.time()
                outputs = model(images)
                end_forward = time.time()

                loss = criterion(outputs.logits, labels)

                start_backward = time.time()
                loss.backward()
                end_backward = time.time()

                optimizer.step()

                avg_time = 0
                iteration += 1

                if (iteration + 1) % tau == 0:
                    # Synchronize the model weights
                    start_sync = time.time()
                    average_weights(model, world_size)
                    end_sync = time.time()
                    avg_time = end_sync - start_sync
                
                # Log the stats
                writer.writerow([epoch, i, loss.item(), end_forward-start_forward, end_backward-start_backward, avg_time, "train"])
                file.flush()

            # Validation step
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    images = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    start_forward = time.time()
                    outputs = model(images)
                    end_forward = time.time()

                    accuracy = val_loss(outputs.logits, labels)
                    # Log the stats
                    writer.writerow([epoch, i, accuracy, end_forward - start_forward, 0, 0, "val"])
            
            file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument("--mode", type=str, choices=["synch", "asynch"], default="synch", 
                        help="Training mode: 'synch' for synchronous or 'asynch' for asynchronous")
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
    parser.add_argument("--interface", type=str, default="eth1",
                        help="Network interface to use for distributed training")
    args = parser.parse_args()
    
    rank, world_size, device = init_distributed()

    mode = args.mode
    model_name = args.model
    minibatch = args.minibatch // world_size # Minibatch size per worker
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    tau = args.tau
    interface = args.interface

    image_processor, model = load_model(model_name, num_labels=101)

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = image_processor(images=list(images), return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs


    train_set, val_set, _ = load_dataset("../data")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=minibatch, sampler=val_sampler, collate_fn=collate_fn)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_name = model_name.split("/")[-1] 

    if mode == "synch":
        synch_train(model, train_loader, val_loader, num_epochs, optimizer, loss, accuracy, device, model_name)
    else:
        measure_memory(asynch_train, model, train_loader, val_loader, num_epochs, tau, optimizer, loss, accuracy, device, model_name)

    clean_up()

