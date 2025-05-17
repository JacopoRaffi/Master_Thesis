from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor
import torchvision
import torch
import torch.distributed as dist
import os

def load_ImageNet(root:str):
    '''Load the ImageNet dataset from the given root directory (both train and test)'''

    train_dataset = torchvision.datasets.ImageNet(root=root, split='train', download=True)
    test_dataset = torchvision.datasets.ImageNet(root=root, split='val', download=True)

    #TODO: return also the validation set
    return train_dataset, test_dataset

def load_model(model_name:str):
    '''Load the ViT model and image processor from Hugging Face Transformers library'''

    # Load the image processor
    if model_name != 'facebook/webssl-mae1b-full2b-224':
        image_processor = ViTImageProcessor.from_pretrained(model_name)
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Load the model
    model = ViTModel.from_pretrained(model_name)

    return image_processor, model

def init_distributed():
    '''Initialize the distributed process group for distributed training'''
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}') if torch.cuda.is_available() else 'cpu'
    dist.init_process_group()

    return rank, world_size, device
