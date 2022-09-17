import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def interest_fixer(interests: str) -> list:
  interests = re.split(", |\n", interests)
  interests = list(dict.fromkeys(list(filter(None, interests))))
  return interests

  
def make_dataset(s_emb, targets, model, target_threshold=None, target_thresholds=None):
    
  # similarity threshold for labelling categories
  # needs to be wayy lower, adding + 2 to len targets to shove the function a bit to the left
  if not target_threshold:
    target_threshold = 1/(2*np.sqrt(len(targets))) 
  
  # getting sbert embs of the target words
  target_embs = model.encode(targets)
  
  # getting cosine similarity between sbert embs and target embs
  # gettings one column of similarity per interest word
  similarity = util.cos_sim(s_emb, target_embs)
  
  # unsure what this does it but works so i dunno hahah
  # ads extra column full of zeroes
  similarity = np.concatenate((np.zeros((similarity.shape[0], 1)), similarity), axis=1) 
  
  # the label is the interest with the highest similarity to the category
  # taking back the column that had the hightest similarity, out of all the interst words, making a 1 column array that has index of highest number per row in the multi similiaty column array, so if the most similiar word is in the third coloun in row 47, then y[46] = 3 (you get 3 because we pad with a bonus 0 column on the left above)
  y = np.argmax(similarity, axis=1) 
  

  
  # here you can use the argmax value in y, to determine what threshold should be put in. You could have a list, and based on y number, take threshold from list and put into other array saying which threshold should be per index, then do the last test at line 31 per index to get tailored thresholds per thing
  
  # here you are getting max value?
  max_sim = np.max(similarity, axis=1)
  
  if target_thresholds:
    
    for i in range(len(y)):
      if max_sim[i] < target_thresholds[y[i] - 1]:
        y[i] = 0
    
    
  else:  
    # if the similarity is lower than the threshold the category is unlabelled
    y[max_sim < target_threshold] = 0 
  
  return y, similarity



def get_interest_ixs(similarity, cluster_ixs, threshold):
  
  # making empty arrays for holding
  interest_ixs = np.zeros(cluster_ixs.shape)
  avg_cluster_similarity = np.zeros(cluster_ixs.shape)
  
  # for every unique cluster label
  for i in np.unique(cluster_ixs):
    cluster_similarity = similarity[cluster_ixs == i, :] 
    cluster_similarity_avg = np.mean(cluster_similarity, 0)
    
    # this is kind of the same as setting y, setting col "index" in the similarity row
    cluster_target = np.argmax(cluster_similarity_avg)
    
    if type(threshold) is list:
      thres_statement = cluster_similarity_avg[cluster_target] < threshold[cluster_target - 1]
    else:
      thres_statement = cluster_similarity_avg[cluster_target] < threshold
      
    # if cluster similiarty is lower than threshold
    if thres_statement: # changed only this to take in list instead of number
      # set interest index 
      interest_ixs[cluster_ixs == i] = 0
      
    else:
      interest_ixs[cluster_ixs == i] = cluster_target
      
    avg_cluster_similarity[cluster_ixs == i] = cluster_similarity_avg[cluster_target]
      
  interest_similarity = np.zeros(interest_ixs.shape)
  
  for i in range(similarity.shape[1]):
    interest_mask = interest_ixs == i
    interest_similarity[interest_mask] = similarity[interest_mask, i]
      
  return interest_ixs.astype(int), interest_similarity, avg_cluster_similarity