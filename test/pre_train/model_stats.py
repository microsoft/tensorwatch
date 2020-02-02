import tensorwatch as tw
import torchvision.models

model_names = ['alexnet', 'resnet18', 'resnet34','densenet121']

for model_name in model_names:
    model = getattr(torchvision.models, model_name)()
    df = tw.model_stats(model, [1, 3, 224, 224])
    print(df)