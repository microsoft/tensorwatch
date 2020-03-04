import copy
import tensorwatch as tw
import torchvision.models
import torch
import time

model = getattr(torchvision.models, 'densenet201')()

def model_timing(model):
    st = time.time()
    for _ in range(20):
        batch = torch.rand([64, 3, 224, 224])
        y = model(batch)
    return time.time()-st

print(model_timing(model))
model_stats = tw.ModelStats(model,  [1, 3, 224, 224], clone_model=False)
print(f'flops={model_stats.Flops}, parameters={model_stats.parameters}, memory={model_stats.inference_memory}')
print(model_timing(model))
