### Part 3-2. Toppic Identification
# conda activate gpuenv
# cd ~/shareWithContainer/SpecializedScience_BERTopic/
# python SS_LLM_GenTopic.py

### import packages
import torch
import pandas as pd
import re
import numpy as np
import math
import glob
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

access_token_read = "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"
login(token = access_token_read)

### model selection - obtain from HuggingFace
# model = "meta-llama/Llama-2-13b-chat-hf"
model = "lmsys/vicuna-13b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(
    model,
    use_auth_token=True,
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

def gen(x, max_length=200):
  sequences = pipeline(
      x,
      do_sample = True,
      top_k = 10,
      num_return_sequences = 1,
      eos_token_id = tokenizer.eos_token_id,
      max_length = max_length,
  )

  return sequences[0]['generated_text'].replace(x, "")

### Get list of files
files = glob.glob('result/freq_*.csv')

### Iterative statments for generating "GenTopic"
for file_name in files:
    print(file_name)
    
    reg_name = re.search(r'freq_(.*).csv', file_name).group(1)
    
    ### Data load & preparation
    dat = pd.read_csv(file_name)
    
    dat = pd.read_csv(files[0])
    dat = dat[dat.Topic != -1] # needed if want to remove outlier
    dat = dat.reset_index()
    dat['GenTopic'] = "" 

    prompt = """.\n List of words above are the outcome of the science publication topic modelling. Generate a new science topic that summarizes them."""

    for i in dat.index:
        dat['GenTopic'][i] = gen(dat['Representation'][i]+prompt, 300)
        
    dat.to_csv("result/freq_"+reg_name+"_gentopic.csv")
