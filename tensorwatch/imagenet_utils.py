# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torchvision import transforms
from . import pytorch_utils
import json, os

def get_image_transform():
    transf = transforms.Compose([ #TODO: cache these transforms?
        get_resize_transform(),
        transforms.ToTensor(),
        get_normalize_transform()
    ])    

    return transf

def get_resize_transform(): 
    return transforms.Resize((224, 224))

def get_normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])   

def image2batch(image):
    return pytorch_utils.image2batch(image, image_transform=get_image_transform())

def predict(model, images, image_transform=None, device=None):
    logits = pytorch_utils.batch_predict(model, images, 
                                         input_transform=image_transform or get_image_transform(), device=device)
    probs = pytorch_utils.logits2probabilities(logits) #2-dim array, one column per class, one row per input
    return probs

_imagenet_labels = None
def get_imagenet_labels():
    # pylint: disable=global-statement
    global _imagenet_labels
    _imagenet_labels = _imagenet_labels or ImagenetLabels()
    return _imagenet_labels

def probabilities2classes(probs, topk=5):
    labels = get_imagenet_labels()
    top_probs = probs.topk(topk)
    # return (probability, class_id, class_label, class_code)
    return tuple((p,c, labels.index2label_text(c), labels.index2label_code(c)) \
        for p, c in zip(top_probs[0][0].data.cpu().numpy(), top_probs[1][0].data.cpu().numpy()))

class ImagenetLabels:
    def __init__(self, json_path=None):
        self._idx2label = []
        self._idx2cls = []
        self._cls2label = {}
        self._cls2idx = {}

        json_path = json_path or os.path.join(os.path.dirname(__file__), 'imagenet_class_index.json')

        with open(os.path.abspath(json_path), "r") as read_file:
            class_json = json.load(read_file)
            self._idx2label = [class_json[str(k)][1] for k in range(len(class_json))]
            self._idx2cls = [class_json[str(k)][0] for k in range(len(class_json))]
            self._cls2label = {class_json[str(k)][0]: class_json[str(k)][1] for k in range(len(class_json))}
            self._cls2idx = {class_json[str(k)][0]: k for k in range(len(class_json))}  

    def index2label_text(self, index):
        return self._idx2label[index]
    def index2label_code(self, index):
        return self._idx2cls[index]
    def label_code2label_text(self, label_code):
        return self._cls2label[label_code]
    def label_code2index(self, label_code):
        return self._cls2idx[label_code]