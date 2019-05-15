# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import scipy.spatial.distance
import heapq
import numpy as np
import torch

def pyt_tensor2np(pyt_tensor):
    if pyt_tensor is None:
        return None
    if isinstance(pyt_tensor, torch.Tensor):
        n = pyt_tensor.data.cpu().numpy()
        if len(n.shape) == 1:
            return n[0]
        else:
            return n
    elif isinstance(pyt_tensor, np.ndarray):
        return pyt_tensor
    else:
        return np.array(pyt_tensor)

def pyt_tuple2np(pyt_tuple):
    return tuple((pyt_tensor2np(t) for t in pyt_tuple))

def pyt_ds2list(pyt_ds, count=None):
    count = count or len(pyt_ds)
    return [pyt_tuple2np(t) for t, c in zip(pyt_ds, range(count))]

def sample_by_class(data, n_samples, class_col=1, shuffle=True):
    if shuffle:
        random.shuffle(data)
    samples = {}
    for i, t in enumerate(data):
        cls = t[class_col]
        if cls not in samples:
            samples[cls] = []
        if len(samples[cls]) < n_samples:
            samples[cls].append(data[i])
    samples = sum(samples.values(), [])
    return samples

def col2array(dataset, col):
    return [row[col] for row in dataset]

def search_similar(inputs, compare_to, algorithm='euclidean', topk=5, invert_score=True):
    all_scores = scipy.spatial.distance.cdist(inputs, compare_to, algorithm)
    all_results = []
    for input_val, scores in zip(inputs, all_scores):
        result = []
        for i, (score, data) in enumerate(zip(scores, compare_to)):
            if invert_score:
                score = 1/(score + 1.0E-6)
            if len(result) < topk:
                heapq.heappush(result, (score, (i, input_val, data)))
            else:
                heapq.heappushpop(result, (score, (i, input_val, data)))
        all_results.append(result)
    return all_results