import torch
import torchvision.models
import tensorwatch as tw

vgg16_model = torchvision.models.vgg16()

drawing = tw.draw_model(vgg16_model, [1, 3, 224, 224])
drawing.save('abc.png')

input("Press any key")