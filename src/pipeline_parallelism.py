import argparse
import torch
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, Schedule1F1B
import time
import csv

from utils import *

from transformers import logging, set_seed
logging.set_verbosity_error()

torch.set_default_device("cpu")
set_seed(seed=42, deterministic=False)