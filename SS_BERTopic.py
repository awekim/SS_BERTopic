### Part 3-1. Toppic Modelling
# conda activate gpuenv
# cd ~/shareWithContainer/SpecializedScience_BERTopic/
# python SS_BERTopic.py

import torch
import pandas as pd
import re
import numpy as np
import math

# Get list of lies
import glob
import re

files = glob.glob('data/regions/reg_pub_sub_m_*.csv')
completed_files = glob.glob('result/freq_ent_*.csv')
completed_files = [filename.replace('result/freq_ent_', 'data/regions/reg_pub_sub_m_') for filename in completed_files]
completed_files = [filename.replace('.csv', '_doc.csv') for filename in completed_files]
remaining_files = [file for file in files if file not in completed_files]
reg_names = [re.search(r'reg_pub_sub_m_(.*)_doc\.csv', path).group(1) for path in remaining_files]

i = 0

for file_name in remaining_files:
    print(file_name)
    
    ### Data load & preparation
    dat = pd.read_csv(file_name)

    reg_name = re.search(r'reg_pub_sub_m_(.*)_doc\.csv', file_name).group(1)

    dat = dat[dat.abstract.isnull()==False]
    dat['abstract'] = [s.replace('\r', '').replace('\n', '').replace('<p>', '').replace('</p>', '') for s in dat['abstract']]
    dat['abstract'] = [re.sub(r'\(C\) 20.*', '', text) for text in dat['abstract']]
    dat['abstract'] = [re.sub(r'\(.*?\)', '', text) for text in dat['abstract']]
    dat['abstract'] = [re.sub(r'\d+', '', text) for text in dat['abstract']] # Remove numbers.
    dat['abstract'] = [re.sub(r"[^\w\s.,']", '', text) for text in dat['abstract']] # Remove all punct excep . & , & '
    dat['abstract'] = [re.sub(r'\s{2,}', '', text) for text in dat['abstract']] # Remove multiple spaces
    dat['abstract'] = [re.sub('Published by Elsevier B.V. All rights reserved.', '', text) for text in dat['abstract']] # Remove multiple spaces
    dat = dat[dat['abstract']!='Editorial Board']
    dat = dat.drop_duplicates()

    docs = dat['abstract'].to_list()

    ### Load BERTopic related packages & initial settings
    from bertopic import BERTopic
    from umap import UMAP
    from sklearn.metrics import silhouette_score
    from hdbscan import HDBSCAN
 
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(stop_words="english")

    from bertopic.vectorizers import ClassTfidfTransformer
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Set Diveristy of Topics (0: no diversity, 1: max diversity)
    from bertopic.representation import MaximalMarginalRelevance
    # representation_model = MaximalMarginalRelevance(diversity=0.5)

    from bertopic.representation import KeyBERTInspired
    # Create your representation model
    representation_model = KeyBERTInspired()
   
    # Set embedding model
    from sentence_transformers import SentenceTransformer
    # Pre-calculate embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, 
                                        show_progress_bar=True)

    # UMAP and HDBSCAN
    umap_model = UMAP(random_state=1004)

    ### Generate BERTopic model
    # We need conditional statement for case with 
    if len(docs) >= 100:
        topic_model = BERTopic(
            min_topic_size = math.floor(len(docs) * 0.03), 
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            calculate_probabilities=True,
            ctfidf_model = ctfidf_model,
            umap_model = umap_model)
    else:
        topic_model = BERTopic(
#             min_topic_size = math.floor(len(docs) * 0.03), 
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            calculate_probabilities=True,
            ctfidf_model = ctfidf_model,
            umap_model = umap_model)
            

    topics, probs = topic_model.fit_transform(docs, embeddings)

    freq = topic_model.get_topic_info()
    freq.to_csv('result/freq_'+reg_names[i] +'.csv', index=False)
    
    topic_model_sum = topic_model.get_document_info(docs)

    topic_model_sum = pd.merge(dat[['pubid','abstract']], 
                                  topic_model_sum.loc[:, ~topic_model_sum.columns.isin(['Name''Representation','Representative_Docs'])], 
                                  left_on='abstract', right_on='Document')

    topic_model_sum.to_csv("result/"+reg_name+"_sumtable.csv")
    topic_model.save("result/"+reg_name+"_bertopic_model")
    
    del topic_model, dat, docs
    
    i = i + 1