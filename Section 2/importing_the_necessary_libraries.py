# Import libraries

import os, sys 
import ipdb # for debugging, variation of pdb
from tqdm import tqdm # A library that allows us to have a progress bar and follow the progress of the training
from datetime import datetime # A library that Allow us to save intermediate checkpoint of our training an add 
# the datetime of our checkpointimport 
import request, zipfile, io #generic libraries



# Pytorch - The leading library to create deep learning and AI

import torch 
import torch.nn as nn
# nn --> stand for neural network torch.nn named as nn
from torch.nn import functional as F

# F.relu(x)        # Just applies max(0, x)
# F.softmax(x)     # Just converts to probabilities  
# F.dropout(x)     # Just randomly turns off neurons
# These are stateless functions - they don't store or learn anything.
# They just take input, apply a mathematical operation, and return output.

# tokenizer - When we are going to train an LLM we are going to use a dataset,
# normal human language, AI can't ingest normal language they work with numbers so we are going to
# split the text in tokens, and sign some numbers to those tokens
import sentencepiece as spm

# these improve performance for Ampere architecture

# In deep learning neural networks, precision refers to the format and accuracy with which numerical values, such as weights and activations, are represented and processed. Integer precision uses whole numbers, while float precision uses decimal points, allowing for more granular representation of values.
# Float precision, especially in formats like float32 or float64, provides higher accuracy and better performance in training complex models, while integer precision can be more efficient in terms of memory and speed for inference on less complex tasks.
# The choice between them impacts model accuracy, computational resources, and performance

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True # Only relevant if you have certain types of GPUs

#Empty GPU Cache Memory
torch.cuda.empty_cache() # Clear GPU memory cache - good practice