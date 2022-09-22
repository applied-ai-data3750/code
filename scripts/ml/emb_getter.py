import pickle

def sbert_emb_getter(data_column, filename='model-cuda'):
  # getting fresh model just in case this helps
  temp_model = pickle.load( open( f"data/{filename}-{device}.pkl", "rb" ) )
  
  # getting the embs here, hoping that this will remove the bloated model object post encodings
  semb = temp_model.encode(data_column)
  
  # returning just the sembs, garbage collector should now clear ram of the unneeded info in the model object that is added post encoding
  return semb