# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def _standardize_data(data, col, whitten, flatten):
    if col is not None:
        data = data[col]

    #TODO: enable auto flattening
    #if data is tensor then flatten it first
    #if flatten and len(data) > 0 and hasattr(data[0], 'shape') and \
    #    utils.has_method(data[0], 'reshape'):

    #    data = [d.reshape((-1,)) for d in data]

    if whitten:
        data = StandardScaler().fit_transform(data)
    return data

def get_tsne_components(data, features_col=0, labels_col=1, whitten=True, n_components=3, perplexity=20, flatten=True, for_plot=True):
    features = _standardize_data(data, features_col, whitten, flatten)
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_results = tsne.fit_transform(features)

    if for_plot:
        comps = tsne_results.tolist()
        labels = data[labels_col]
        for i, item in enumerate(comps):
            item.append(None) # annotation
            item.append(str(labels[i])) # text
            item.append(labels[i]) # color
        return comps
    return tsne_results



