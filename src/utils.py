from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, ViTForImageClassification
import torchvision
import torch
import torch.distributed as dist
import os
import psutil

def get_memory_usage():
    """Get the current memory usage of the process."""
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    return mem

def accuracy(logits, labels):
    """Compute the accuracy of the model predictions."""
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    acc = correct / len(labels)
    return acc

def load_dataset(root:str, val_perc:float=0.2):
    '''Load the ImageNet dataset from the given root directory (both train and test)'''

    dev_dataset = torchvision.datasets.Food101(root=root, split="train", download=True)
    test_dataset = torchvision.datasets.Food101(root=root, split="test", download=True)

    # Split the training dataset into train and validation sets
    num_train = len(dev_dataset)
    indices = list(range(num_train))
    split = int(val_perc * num_train)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset = torch.utils.data.Subset(dev_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dev_dataset, val_indices)

    return train_dataset, val_dataset, test_dataset

def load_model(model_name:str, num_labels:int):
    '''Load the ViT model and image processor from Hugging Face Transformers library'''

    # Load the image processor
    if model_name != "facebook/webssl-mae1b-full2b-224":
        image_processor = ViTImageProcessor.from_pretrained(model_name)
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Load the model
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)

    return image_processor, model

def init_distributed():
    '''Initialize the distributed process group for distributed training'''
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else "cpu"
    dist.init_process_group()

    return rank, world_size, device
