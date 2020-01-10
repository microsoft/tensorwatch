import tensorwatch as tw
import torchvision.models

alexnet_model = torchvision.models.alexnet()

df = tw.model_stats(alexnet_model, [1, 3, 224, 224])
print(df)