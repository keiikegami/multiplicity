import pandas as pd
import numpy as np
import generatedata as gd
from sklearn.cluster import KMeans

def cluster(df):
    # data
    X = df.drop(["entryprob1", "entryprob2", "equitype","single"], axis = 1)
    
    # cluster num = 3
    k_means = KMeans(n_clusters = 3)
    k_means.fit(X)
    df["k-means-pred"] = pd.DataFrame(k_means.labels_)
    
    # cluster num = 2
    k_means2 = KMeans(n_clusters = 2)
    k_means2.fit(X)
    df["k-means2-pred"] = pd.DataFrame(k_means2.labels_)
    
    return df
    