from tensorwatch.saliency import saliency
from tensorwatch import image_utils, imagenet_utils, pytorch_utils

model = pytorch_utils.get_model('resnet50')
raw_input, input, target_class = pytorch_utils.image_class2tensor('../data/test_images/dogs.png', 240,  #'../data/elephant.png', 101,
    image_transform=imagenet_utils.get_image_transform(), image_convert_mode='RGB')

results = saliency.get_image_saliency_results(model, raw_input, input, target_class)
figure = saliency.get_image_saliency_plot(results)

image_utils.plt_loop()





