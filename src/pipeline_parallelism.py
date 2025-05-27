import argparse
import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, Schedule1F1B, build_stage
import time
import csv

from utils import *

from transformers import logging, set_seed
logging.set_verbosity_error()

torch.set_default_device("cpu")
set_seed(seed=42, deterministic=False)

def split_model(model, num_stages, input_sample, device="cpu"):
    num_block = len(model.vit.encoder.layer)
    blocks_per_stage = num_block // num_stages

    keys = [f"vit.encoder.layer.{i}" for i in range(blocks_per_stage, num_block, blocks_per_stage)]

    split_spec = {}
    for key in keys:
        split_spec[key] = SplitPoint.BEGINNING # each stage has num_blocs//num_stages blocks

    pipe = pipeline(model, mb_args=(input_sample, ), split_spec=split_spec)
    
    stage_mode = pipe.get_stage_module(dist.get_rank()) # get the submodule for the current rank
    stage = build_stage(stage_mode, dist.get_rank(), pipe.info(), device) # build the stage for the Scheduler class

    return stage

def train(stage, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, n_microbatch:int,
                num_epochs:int, optimizer:torch.optim, train_loss:torch.nn, val_loss:callable, device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    
    train_schedule = Schedule1F1B(stage, n_microbatches=n_microbatch, loss_fn=train_loss)
    val_schedule = Schedule1F1B(stage, n_microbatches=n_microbatch)

    # csv file to save the stats
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = train_loader.batch_size
    file_name = f"../log/pp/rank_{rank}_pp_{world_size}_minibatch_{batch_size}_{model_name}_model.csv"

    stage_index = stage.stage_index

    with open(file_name, mode="w+") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "batch_id", "loss", "minibatch_time", "phase"])

        for epoch in range(num_epochs):
            stage.submodule.train()  # Set the stage to training mode
            for i, batch in enumerate(train_loader):
                images = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                start_time = time.time()

                if stage_index == 0: # First stage
                    train_schedule.step(images)
                    loss = 0
                elif stage_index == (dist.get_world_size() - 1): # Last stage
                    output = train_schedule.step(labels)
                    loss = train_loss(output.logits, labels).item()
                else: # Intermediate stages
                    train_schedule.step()
                    loss = 0

                end_time = time.time()
                
                optimizer.step()

                # Log the stats
                writer.writerow([epoch, i, loss.item(), end_time-start_time, "train"])
                file.flush()

            # Validation step
            stage.submodule.eval()  # Set the stage to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    images = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    start_time = time.time()

                    if stage_index == 0: # First stage
                        train_schedule.step(images)
                        loss = 0
                    elif stage_index == (dist.get_world_size() - 1): # Last stage
                        output = train_schedule.step()
                        loss = val_loss(output.logits, labels).item()
                    else: # Intermediate stages
                        train_schedule.step()
                        loss = 0

                    end_time = time.time()

                    # Log the stats
                    writer.writerow([epoch, i, loss, end_time - start_time, "val"])
            
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
    parser.add_argument("--microbatch", type=int, default=5, 
                        help="Size of the microbatch for pipeline parallelism")
    args = parser.parse_args()

    rank, world, device = init_distributed()

    model_name = args.model
    minibatch = args.minibatch
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    microbatch = args.microbatch

    image_processor, model = load_model("google/vit-base-patch16-224-in21k", 101)
    stage = split_model(model, num_stages=4, input_sample=torch.randn(1, 3, 224, 224))

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = image_processor(images=list(images), return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    train_set, val_set, _ = load_dataset("../data")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=minibatch, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model_name = model_name.split("/")[-1] 

    measure_memory(train, stage, train_loader, val_loader, num_epochs, optim, criterion, accuracy, device, model_name)

    clean_up()