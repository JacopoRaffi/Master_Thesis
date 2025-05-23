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

def get_memory_usage():
    """Get the current memory usage of the process."""
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    return mem

def synch_train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                num_epochs:int, optimizer:torch.optim, train_loss:torch.nn, val_loss:callable, device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    

    model.to(device)

    # csv file to save the stats
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = train_loader.batch_size * world_size
    file_name = f"../log/rank_{rank}_synch_ddp_{world_size}_minibatch_{batch_size}_{model_name}_model.csv"

    model = DDP(model)

    with open(file_name, mode="w+") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "batch_id", "loss", "forward_time", "backward_time", "peak_memory_usage(MB)", "phase"])

        for epoch in range(num_epochs):
            model.train()
            for i, batch in enumerate(train_loader):
                images = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                start_forward = time.time()
                outputs = model(images)
                end_forward = time.time()

                # Get the memory usage after forward pass
                mem_usage_forward = get_memory_usage()

                loss = train_loss(outputs.logits, labels)

                start_backward = time.time()
                loss.backward()
                end_backward = time.time()
                
                optimizer.step()

                # Log the stats
                writer.writerow([epoch, i, loss.item(), end_forward - start_forward, end_backward - start_backward, mem_usage_forward, "train"])
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

                    # Get the memory usage after forward pass
                    mem_usage_forward = get_memory_usage()

                    accuracy = val_loss(outputs.logits, labels)
                    # Log the stats
                    writer.writerow([epoch, i, accuracy, end_forward - start_forward, 0, mem_usage_forward, "val"])
            
            file.flush()

def asynch_train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader,
                  num_epochs:int, asynch_iterations:int, optimizer:torch.optim, criterion:torch.nn, val_loss:callable):
    pass


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
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for the optimizer")
    parser.add_argument("--asynch_iterations", type=int, default=10,
                        help="Number of iteration to be performed before synchronizing")
    args = parser.parse_args()
    
    rank, world_size, device = init_distributed()

    mode = args.mode
    model_name = args.model
    minibatch = args.minibatch // world_size # Minibatch size per worker
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay

    image_processor, model = load_model(model_name, num_labels=101)

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = image_processor(images=list(images), return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs


    train_set, val_set, test_set = load_dataset("../data")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=minibatch, collate_fn=collate_fn)
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=minibatch)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_name = model_name.split("/")[-1] 

    if mode == "synch":
        synch_train(model, train_loader, val_loader, num_epochs, optimizer, loss, accuracy, model_name=model_name)
    else:
        asynch_train()

