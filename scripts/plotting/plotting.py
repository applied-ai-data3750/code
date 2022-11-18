import numpy as np
import pandas as pd
from typing import Tuple
import time

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .topic_finding import *

# when you actually cast the type here, then it works with how pandas casts types and you don't have to worry about copying seriers
def result_df_maker(embeddings: np.ndarray, cluster_labels: np.ndarray, titles: np.ndarray, bonus_words=None, interest_words=None) -> pd.DataFrame:
  """
  Function to make a dataframe with the embeddings, cluster labels, topic per cluster label and titles.

  Args:
      embeddings (np.ndarray): 2D array of embeddings.
      cluster_labels (np.ndarray): array of cluster labels.
      titles (np.ndarray): array of titles.

  Returns:
      pd.DataFrame: Dataframe with embeddings, cluster labels, group_topics per cluster, and titles.
  """
  result = pd.DataFrame(embeddings, columns=['x', 'y'])

  result["titles"] = titles

  result["cluster_label"] = cluster_labels

  if interest_words: result["interest_words"] = interest_words

  topic_dict = topic_by_clusterId(text=result["titles"].to_numpy(), cluster_label=result["cluster_label"].to_numpy(), bonus_words=bonus_words, interest_words=interest_words)

  if interest_words:
    result["group_topics"] = result["interest_words"].apply(lambda x: topic_dict[x])

  else:
    result["group_topics"] = result["cluster_label"].apply(lambda x: topic_dict[x])


  result["group_topics"] = result["group_topics"].apply(lambda x: " ".join(x))

  return result

def result_splitter(result: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
  """
  Function to split the dataframe into two dataframes, one for clustered and one for outliers.

  Args:
      result (pd.DataFrame): Dataframe with embeddings, cluster labels, group_topics per cluster, and titles.

  Returns:
      Tuple[np.ndarray, np.ndarray]: Tuple of two dataframes, one for clustered and one for outliers.
  """

  clustered = result.loc[result.cluster_label != -1, :]
  outliers = result.loc[result.cluster_label == -1, :]
  return clustered, outliers

# the cavalry is not here, but it's fine! Why? I am here!
def result_tracer(clustered: pd.DataFrame, outliers: pd.DataFrame, interest_words=None) -> Tuple[go.Scattergl, go.Scattergl]:
  """
  Function to make a scatter traces of the clustered and outliers.

  Args:
      clustered (pd.DataFrame): clustered dataframe to be colored by cluster and get hover data
      outliers (pd.DataFrame): outlier data frame with grey color and no hover data

  Returns:
      Tuple[go.Scattergl, go.Scattergl]: Tuple of two scatter traces.
  """



  if not interest_words:
    hover_template = "<b>Commit Msg:</b> %{customdata[0]} <br><b>Group Topics:</b> %{customdata[1]} <br><b>Cluster Id:</b> %{customdata[2]}<extra></extra>"
    custom_data = np.column_stack([clustered.titles, clustered.group_topics, clustered.cluster_label])
    clustered["color_col"] = clustered.cluster_label

  else:
    cluster_id_dict = {}
    for i, word in enumerate(set(list(clustered.interest_words))):
      cluster_id_dict[word] = i

    clustered["interest_id"] = clustered.interest_words.apply(lambda x: cluster_id_dict[x])

    hover_template = "<b>Commit Msg:</b> %{customdata[0]} <br><b>Traget Interest:</b> %{customdata[1]} <br><b>Group Topics:</b> %{customdata[2]} <br><b>Cluster Id:</b> %{customdata[3]}<extra></extra>"
    custom_data = np.column_stack([clustered.titles, clustered.interest_words, clustered.group_topics, clustered.cluster_label])
    clustered["color_col"] = clustered.interest_id

  trace_cluster = go.Scattergl(
    x=clustered.x, 
    y=clustered.y, 
    mode="markers", 
    name="Clustered",

    # styling markers
    marker=dict(
      size=2, 
      color=clustered.color_col,
      colorscale="Rainbow"
    ), 

    # setting hover text to the titles of the videos
    hovertemplate=hover_template, 
    customdata=custom_data,
  )

  trace_outlier = go.Scattergl(
    x=outliers.x,
    y=outliers.y,
    mode="markers",
    name="Outliers",

    marker=dict(
      size=1,
      color="grey"
    ),

    hovertemplate="Outlier<extra></extra>"
  )

  return trace_cluster, trace_outlier

def result_tracer_wrapper(uembs: np.ndarray, cluster_labels: np.ndarray, titles: np.ndarray, bonus_words=None, interest_words=None) -> Tuple[go.Scattergl, go.Scattergl]:
  """
  Function to make a scatter traces of the clustered and outliers.

  Args:
      uembs (np.ndarray): 2D array of embeddings.
      cluster_labels (np.ndarray): array of cluster labels.
      titles (np.ndarray): array of titles.

  Returns:
      Tuple[go.Scattergl, go.Scattergl]: Tuple of two scatter traces.
  """

  result = result_df_maker(uembs, cluster_labels, titles, bonus_words=bonus_words, interest_words=interest_words)
  
  # if not interest_words:
  #   clustered, outliers = result_splitter(result)
  # else:
  #   clustered = result
  #   outliers = pd.DataFrame(columns=result.columns)

  clustered, outliers = result_splitter(result)


  trace_cluster, trace_outlier = result_tracer(clustered, outliers, interest_words=interest_words)
  
  return trace_cluster, trace_outlier

def subplotter(trace_nested_list: list, titles: list, base_size=1000) -> go.Figure:
    """
    Function to make a figure with subplots of the clustered and outliers.

    Args:
        trace_nested_list (list): list holding rows of columns, each column holding traces. 
        titles (list): Titles for the subplots
        base_size (int, optional): Base size of the sub plots. Defaults to 1000.

    Returns:
        go.Figure: Figure with subplots.
    """
    
    row_count = len(trace_nested_list)
    col_count = len(trace_nested_list[0])
    
    fig = make_subplots(
        rows=row_count, 
        cols=col_count,
        subplot_titles=(titles),
        vertical_spacing=0.02,
        horizontal_spacing=0.02
    )

    for i, row in enumerate(trace_nested_list):
        for j, col in enumerate(row):

            # adding both outlieers and clustered
            for trace in col:
                fig.add_trace(trace, row=i+1, col=1)
    
    # figure settings
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    fig.update_layout(width=base_size*col_count, height=base_size*row_count, plot_bgcolor='rgba(250,250,250,1)')

    return fig

def fig_show_save(fig: go.Figure, filename: str, show=True):
  """
  Function to show and save a figure.

  Args:
      fig (go.Figure): fig to be saved and shown
      filename (str): filename to save the figure, without extension
      show (bool, optional): Option to disable showing of figure 
      (in case too big for notebook). Defaults to True.
  """
  # adding directory and timestamp to filename
  filename_newest = f"figures/newest-{filename}"
  
  filename_time = f"figures/{int(time.time())}-{filename}"


  
  # writing both interactible .html and static image .png
  fig.write_html(f"{filename_time}.html")
  fig.write_image(f"{filename_time}.png")

  # also overwriting the "newest" image files to make easy to keep readme up to date
  fig.write_html(f"{filename_newest}.html")
  fig.write_image(f"{filename_newest}.png")

  if show: 
    fig.show()
  else:
    return fig