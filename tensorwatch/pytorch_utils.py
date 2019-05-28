# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torchvision import models, transforms
import torch
import torch.nn.functional as F
from . import utils, image_utils
import os

def get_model(model_name):
    model = models.__dict__[model_name](pretrained=True)
    return model

def tensors2batch(tensors, preprocess_transform=None):
    if preprocess_transform:
        tensors = tuple(preprocess_transform(i) for i in tensors)
    if not utils.is_array_like(tensors):
        tensors = tuple(tensors)
    return torch.stack(tensors, dim=0)

def int2tensor(val):
    return torch.LongTensor([val])

def image2batch(image, image_transform=None):
    if image_transform:
        input_x = image_transform(image)
    else: # if no transforms supplied then just convert PIL image to tensor
        input_x = transforms.ToTensor()(image)
    input_x = input_x.unsqueeze(0) #convert to batch of 1
    return input_x

def image_class2tensor(image_path, class_index=None, image_convert_mode=None, 
                       image_transform=None):
    image_pil = image_utils.open_image(os.path.abspath(image_path), convert_mode=image_convert_mode)
    input_x = image2batch(image_pil, image_transform)
    target_class = int2tensor(class_index) if class_index is not None else None
    return image_pil, input_x, target_class

def batch_predict(model, inputs, input_transform=None, device=None):
    if input_transform:
        batch = torch.stack(tuple(input_transform(i) for i in inputs), dim=0)
    else:
        batch = torch.stack(inputs, dim=0)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    batch = batch.to(device)
    
    outputs = model(batch)

    return outputs

def logits2probabilities(logits):
    return F.softmax(logits, dim=1)

def tensor2numpy(t):
    return t.data.cpu().numpy()
