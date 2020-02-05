import tensorwatch as tw
import torchvision.models

model_names = ['alexnet', 'resnet18', 'resnet34', 'resnet101', 'densenet121']

for model_name in model_names:
    model = getattr(torchvision.models, model_name)()
    model_stats = tw.ModelStats(model,  [1, 3, 224, 224], clone_model=False)
    print(f'{model_name}: flops={model_stats.Flops}, parameters={model_stats.parameters}, memory={model_stats.inference_memory}')