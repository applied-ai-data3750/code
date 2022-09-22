### Slicing data timewise

df_slices = []

step = len(df)//20

for i in range(1, 21):
    df_slices.append(df.iloc[(i-1)*step:i*step:])

print(len(df_slices))
print(type(df_slices[0]))

df_slices[-1]


for slice in df_slices:
    print(slice["commit"].head(1))
    print(slice["commit"].tail(1))

fig_list = []

for i, slice in enumerate(df_slices):

    # unsupervised
    slice_embs = sbert_emb_getter(slice["subject_clean"].to_numpy(), device=device)
    slice_uembs = UMAP(n_neighbors=20, min_dist=0.1).fit_transform(slice_embs)
    slice_clusters_2d = HDBSCAN(min_cluster_size=100, cluster_selection_method="leaf").fit(slice_uembs)

    slice_trace_cluster_2d, slice_trace_outlier_2d = result_tracer_wrapper(
        slice_uembs, 
        slice_clusters_2d.labels_, 
        slice["subject_clean"].to_numpy(), 
        bonus_words=interests
    )

    s_col = [slice_trace_cluster_2d, slice_trace_outlier_2d]

    s_row = [s_col]

    fig = subplotter([s_row, ], [f"Timeline slice {i}", ])

    fig_list.append(fig)

filename_timeline = "figures/animation/timeline"

for i, fig in enumerate(fig_list):
    fig.write_html(f"{filename_timeline}-{i}.html")
    fig.write_image(f"{filename_timeline}-{i}.png")


"""
clustre først, sette topics, så slice, for å enklere tracke topics og cluster over tid

så sette samme område / axis på layout, sånn at clusters ser bra ut, / håp om consistent stuff

vi kan ta alle embeddings i samme "plot" men ha egene traces slices basert på tid, sånn at de står på en sensible måte i forhold til hverandre
"""