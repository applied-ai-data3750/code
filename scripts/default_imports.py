# regex
import re

# pandas + numpy
import numpy as np
import pandas as pd

# setting pandas options
pd.set_option('display.max_colwidth', 200)


# storing and loading models
import pickle

# to set types for functions
from typing import Tuple

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# gpu debug
import torch

# setting device to use GPU for NLP backend if you have GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"


# SBERT
from sentence_transformers import SentenceTransformer

# UMAP
from umap import UMAP

#HDBSCAN
from hdbscan import HDBSCAN

# topic finding
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading model from pickle if possible, to avoid downloading it again
try:
    model = pickle.load(open(f'data/model-{device}.pkl', 'rb'))

    model_load = True

except:
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    pickle.dump(model, open(f'data/model-{device}.pkl', 'wb'))

    model_load = False

print(f"""
GPUs detected:          {torch.cuda.device_count()}
Using GPU:              {torch.cuda.is_available()}
Device:                 {device}
Got model from pickle:  {model_load}
""")