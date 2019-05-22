# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from receptivefield.pytorch import PytorchReceptiveField
#from receptivefield.image import get_default_image
import numpy as np

def _get_rf(model, sample_pil_img):
    # define model functions
    def model_fn():
        model.eval()
        return model

    input_shape = np.array(sample_pil_img).shape

    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape=input_shape)
    return rf, rf_params

def plot_receptive_field(model, sample_pil_img, layout=(2, 2), figsize=(6, 6)):
    rf, rf_params = _get_rf(model, sample_pil_img) # pylint: disable=unused-variable
    return rf.plot_rf_grids(
        custom_image=sample_pil_img, 
        figsize=figsize, 
        layout=layout)

def plot_grads_at(model, sample_pil_img, feature_map_index=0, point=(8,8), figsize=(6, 6)):
    rf, rf_params = _get_rf(model, sample_pil_img) # pylint: disable=unused-variable
    return rf.plot_gradient_at(fm_id=feature_map_index, point=point, image=None, figsize=figsize)
