# regex, time, Tuple return type, pickle for object dumping
import re
import time
from typing import Tuple
import pickle

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# data handling
import numpy as np
import pandas as pd

# setting pandas options
pd.set_option('display.max_colwidth', 200)


# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# gpu debug
import torch

# setting device to use GPU for NLP backend if you have GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ML
from sentence_transformers import SentenceTransformer, util
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer



# stopwords to automate semi supervised topic finding
import nltk
from nltk.corpus import stopwords

try:
    # seeing if stopwords are already downloaded
    nltk.data.find('corpora/stopwords')
    local_stopwords = True
except:
    # downloading stopwords if not already downloaded
    nltk.download('stopwords')
    local_stopwords = False



# Loading model from pickle if possible, to avoid downloading it again
try:
    model = pickle.load(open(f'data/model-{device}.pkl', 'rb'))

    model_load = True

except:
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    pickle.dump(model, open(f'data/model-{device}.pkl', 'wb'))

    model_load = False

print(f"""
Local stopwords:        {local_stopwords}
GPUs detected:          {torch.cuda.device_count()}
Using GPU:              {torch.cuda.is_available()}
Device:                 {device}
Got model from pickle:  {model_load}
""")