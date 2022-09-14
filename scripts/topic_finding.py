from scripts.default_imports import *

def tfidf_most_relevant_word(input: list, num_words=5) -> list:
  """
  Function that finds the most relevant words per cluster id.

  Args:
      input (list): A list of title strings aggregated by cluster id.
      num_words (int, optional): How many words you want. Defaults to 5.

  Returns:
      list: Returns a list of most relevant words, with lenght of unique cluster Ids
  """

  most_relevant_words = []
  
  for corpus in input:
        
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    
    importance = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(vectorizer.get_feature_names_out()) # get_feature_names
    most_relevant_words.append(tfidf_feature_names[importance[:num_words]])

  return most_relevant_words

def topic_by_clusterId(result: pd.DataFrame) -> dict:
  """
  Function that maps topics to cluster ids.

  Args:
      result (pd.DataFrame): Dataframe with cluster ids and topics.

  Returns:
      dict: Dictionary with cluster ids as keys and topics as values.
  """

  #print(result.isna().sum())

  df_group = result[["titles", "cluster_label"]].groupby("cluster_label").agg(list).reset_index()

  df_group["topics"] = tfidf_most_relevant_word(df_group["titles"])

  return dict(zip(df_group.cluster_label, df_group.topics))