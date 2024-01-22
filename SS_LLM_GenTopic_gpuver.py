### Part 3-2. Toppic Identification
# conda activate gpuenv
# cd ~/shareWithContainer/SpecializedScience_BERTopic/
# python SS_LLM_GenTopic_gpuenv.py

### import packages
import torch
import pandas as pd
import re
import numpy as np
import math
import glob
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from huggingface_hub import login

### 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("D:/LLM/vicuna-13b-v1.5", use_auth_token=True)
model = LlamaForCausalLM.from_pretrained("D:/LLM/vicuna-13b-v1.5",
    torch_dtype=torch.float16)

model.to(device)

### Get list of files
files = glob.glob('SS_BERTopic_results/freq_*.csv')

### Iterative statments for generating "GenTopic"
for file_name in files:
    print(file_name)

    reg_name = re.search(r'freq_(.*).csv', file_name).group(1)

    ### Data load & preparation
    dat = pd.read_csv(file_name)

    dat = dat[dat.Topic != -1] # needed if want to remove outlier
    dat = dat.reset_index()
    dat['GenTopic'] = "" 

    prompt = """.\n List of words above are the outcome of the science publication topic modelling. Generate a new science topic that summarizes them as follow:\n topic: <new science topic>"""

    for i in dat.index:
        inputs = tokenizer(dat['Representation'][i]+prompt, return_tensors='pt')
        inputs.to(device)
        generate_ids = model.generate(inputs.input_ids, max_length = 150)
        dat['GenTopic'][i] = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        dat['GenTopic'][i] = dat['GenTopic'][i].replace(dat['Representation'][i], "")
        dat['GenTopic'][i] = dat['GenTopic'][i].replace(prompt, "")
            
    dat.to_csv("SS_BERTopic_results/freq_"+reg_name+"_gentopic.csv")

