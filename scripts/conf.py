conf = {

    # to signal if there is new data
    "fresh_data" : True,

    # if you want to show figures when running the script
    "show_figs": True,

    # ammount of interest words for semi supervised
    "interest_words": 20,

    # generate interests or use static list
    "generate_interests": True,

    # fresh embeddings and fresh dimensionality reduction
    "fresh_embs": False,
    "fresh_uembs": False,

    # fresh supervised embeddings
    "fresh_s_embs": False,
    "fresh_s_uembs": False
    
}

# if there is new data, then we need to get new embeddings and new dimensionality reduction
if conf["fresh_data"]:
    conf["fresh_embs"] = True
    conf["fresh_uembs"] = True
    conf["fresh_s_embs"] = True
    conf["fresh_s_uembs"] = True

names = {
    "model-cpu" : "model-cpu.pkl",
    "model-cuda" : "model-cuda.pkl",

    "bonus-words": "bonus-words.pkl",
    "target-interest-list": "target-interest-list.pkl",
    "df" : "df.pkl",
    "dfres" : "dfres.pkl",
    
    "embs-cuda" : "embs-cuda.pkl",
    "embs-cpu" : "embs-cpu.pkl",
    
    "uembs-cuda" : "uembs-cuda.pkl",
    "uembs-cpu" : "uembs-cpu.pkl",
    "uembs-s-cuda" : "uembs-s-cuda.pkl",
    "uembs-s-cpu" : "uembs-s-cpu.pkl",
}

for k, v in names.items():
    names[k] = f"data/{v}"