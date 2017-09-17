import pandas as pd
import numpy as np
import generatedata as gd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# unsupervised clustering (k-means)
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
    
# entry probability prediction
# return DataFrame with estimated probability of entry
def entry_predict(data):
    
    # variables
    X1 = data.drop(["entryprob1", "entryprob2", "equitype","single", "entry1", "entry2"], axis = 1)
    X2 = data.drop(["entryprob1", "entryprob2", "equitype","single",  "entry1", "entry2"], axis = 1)
    y1 = data.entry1
    y2 = data.entry2

    clf1 = LogisticRegression()
    clf1.fit(X1, y1)
    coeff1 = pd.DataFrame([X1.columns, clf1.coef_[0]]).T
    data["logit_entry1"] = clf1.predict_proba(X1)[:, 1]
    
    clf2 = LogisticRegression()
    clf2.fit(X2, y2)
    coeff2 = pd.DataFrame([X2.columns, clf2.coef_[0]]).T
    data["logit_entry2"] = clf2.predict_proba(X2)[:, 1]
    
    model1 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    model1 = model1.fit(X1, y1)
    data["poly_entry1"] = model1.predict(X1)
    
    model2 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    model2 = model2.fit(X2, y2)
    data["poly_entry2"] = model2.predict(X2)
    
    return data

# scatter : "real entry probability vs logit "and "real entry probability vs polynomial"
# have to do "entry_predict" before this
def scatter(data):
    probs1 = df1.loc[:, ["entryprob1", "logit_entry1", "poly_entry1"]]
    probs2 = df1.loc[:, ["entryprob2", "logit_entry2", "poly_entry2"]]

    fig, axes = plt.subplots(2, len(probs1.columns.values)-1, sharey=True, figsize=(20, 20))

    for i, col in enumerate(probs1.columns.values[1:]):
        probs1.plot(x = [col], y = ["entryprob1"], kind="scatter", ax=axes[0, i], grid = True)

    for i, col in enumerate(probs2.columns.values[1:]):
        probs2.plot(x = [col], y = ["entryprob2"], kind="scatter", ax=axes[1, i], grid = True)

    plt.show()