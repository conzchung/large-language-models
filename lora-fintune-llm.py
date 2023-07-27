import os
os.environ['CUDA-VISIBLE_DEVICE'] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    load_in_8bit=True,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

