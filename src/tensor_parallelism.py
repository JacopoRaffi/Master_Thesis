import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor import DTensor
import torch.distributed as dist
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
import time
import psutil
import csv

from utils import *

from transformers import logging, set_seed
logging.set_verbosity_error()

torch.set_default_device("cpu")
set_seed(seed=42, deterministic=False)

def parallelize_vit_model(model, tp_mesh):
    for _, block in enumerate(model.vit.encoder.layer):
        layer_tp_plan = {
            # Attention qkv (Colwise)
            "attention.attention.query": ColwiseParallel(),
            "attention.attention.key": ColwiseParallel(),
            "attention.attention.value": ColwiseParallel(),

            # Attention output (Rowwise)
            "attention.output.dense": RowwiseParallel(),

            # Feedforward (Colwise -> Rowwise)
            "intermediate.dense": ColwiseParallel(),
            #"output.dense": RowwiseParallel(),
        }

        attn = block.attention.attention
        attn.num_attention_heads = attn.num_attention_heads // tp_mesh.size()
        attn.all_head_size = attn.all_head_size // tp_mesh.size()

        parallelize_module(
            block,
            tp_mesh,
            layer_tp_plan,
        )

    model = parallelize_module(
        model,
        tp_mesh,
        {
            "classifier": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    return model

def train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                num_epochs:int, optimizer:torch.optim, train_loss:torch.nn, val_loss:callable, device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    
    model.to(device)

    base_memory_usage = get_memory_usage()

    # csv file to save the stats
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = train_loader.batch_size
    file_name = f"../log/tp/rank_{rank}_tp_{world_size}_minibatch_{batch_size}_{model_name}_model.csv"

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
                writer.writerow([epoch, i, loss.item(), end_forward-start_forward, end_backward-start_backward, mem_usage_forward, "train"])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensor Parallelism")
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
    args = parser.parse_args()

    rank, world, device = init_distributed()

    model_name = args.model
    minibatch = args.minibatch
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay

    image_processor, model = load_model(model_name, num_labels=101)

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = image_processor(images=list(images), return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device_name, (world,)) #1D Parallelism

    model = parallelize_vit_model(model, mesh)

    train_set, val_set, _ = load_dataset("../data")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=minibatch, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model_name = model_name.split("/")[-1] 

    train(model, train_loader, val_loader, num_epochs, optim, criterion, accuracy, device=device, model_name=model_name)

    clean_up()
    

