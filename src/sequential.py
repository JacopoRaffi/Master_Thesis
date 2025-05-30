import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import csv
from utils import *

torch.set_default_device("cpu")

torch.set_num_threads(28)
torch.set_num_interop_threads(28)

def accuracy(logits, labels):
    """Compute the accuracy of the model predictions."""
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    acc = correct / len(labels)
    return acc

def train(model:torch.nn, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
                num_epochs:int, optimizer:torch.optim, train_loss:torch.nn, val_loss:callable, 
                device:str='cpu', model_name:str='vit-base-patch16-224-in21k'):
    

    model.to(device)
    # csv file to save the stats
    batch_size = train_loader.batch_size
    file_name = f"../log/seq_minibatch_{batch_size}_model_{model_name}_1.csv"

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

                loss = train_loss(outputs.logits, labels)
                accuracy = val_loss(outputs.logits, labels)
                end_forward = time.time()

                start_backward = time.time()
                loss.backward()
                end_backward = time.time()
                optimizer.step()

                # Log the stats
                writer.writerow([epoch, i, loss.item(), end_forward - start_forward, end_backward - start_backward, "train"])
                file.flush()

            # Validation step
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    images = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)
                    start_forward = time.time()
                    outputs = model(images)
                    accuracy = val_loss(outputs.logits, labels)
                    end_forward = time.time()

                    # Log the stats
                    writer.writerow([epoch, i, accuracy, end_forward - start_forward, 0, "val"])
        
            file.flush()
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Training Script")
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k",
                        help="Name of the ViT model to use")
    parser.add_argument("--minibatch", type=int, default=256, 
                        help="Size of the minibatch for training")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for the optimizer")
    args = parser.parse_args()

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


    train_set, val_set, test_set = load_dataset("../data")    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=minibatch, collate_fn=collate_fn)
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=minibatch, collate_fn=collate_fn)
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_name = model_name.split("/")[-1] 

    train(model, train_loader, val_loader, num_epochs, optimizer, loss, accuracy, device="cpu", model_name=model_name)

