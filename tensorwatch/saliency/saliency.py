from .gradcam import GradCAMExplainer
from .backprop import VanillaGradExplainer, GradxInputExplainer, SaliencyExplainer, \
    IntegrateGradExplainer, DeconvExplainer, GuidedBackpropExplainer, SmoothGradExplainer
from .deeplift import DeepLIFTRescaleExplainer
from .occlusion import OcclusionExplainer
from .epsilon_lrp import EpsilonLrp
from .lime_image_explainer import LimeImageExplainer, LimeImagenetExplainer
import skimage.transform
import torch
import math
from .. import image_utils

class ImageSaliencyResult:
    def __init__(self, raw_image, saliency, title, saliency_alpha=0.4, saliency_cmap='jet'):
        self.raw_image, self.saliency, self.title = raw_image, saliency, title
        self.saliency_alpha, self.saliency_cmap = saliency_alpha, saliency_cmap

def _get_explainer(explainer_name, model, layer_path=None):
    if explainer_name == 'gradcam':
        return GradCAMExplainer(model, target_layer_name_keys=layer_path, use_inp=True)
    if explainer_name == 'vanilla_grad':
        return VanillaGradExplainer(model)
    if explainer_name == 'grad_x_input':
        return GradxInputExplainer(model)
    if explainer_name == 'saliency':
        return SaliencyExplainer(model)
    if explainer_name == 'integrate_grad':
        return IntegrateGradExplainer(model)
    if explainer_name == 'deconv':
        return DeconvExplainer(model)
    if explainer_name == 'guided_backprop':
        return GuidedBackpropExplainer(model)
    if explainer_name == 'smooth_grad':
        return SmoothGradExplainer(model)
    if explainer_name == 'deeplift':
        return DeepLIFTRescaleExplainer(model)
    if explainer_name == 'occlusion':
        return OcclusionExplainer(model)
    if explainer_name == 'lrp':
        return EpsilonLrp(model)
    if explainer_name == 'lime_imagenet':
        return LimeImagenetExplainer(model)

    raise ValueError('Explainer {} is not recognized'.format(explainer_name))

def _get_layer_path(model):
    if model.__class__.__name__ == 'VGG':
        return ['features', '30'] # pool5
    elif model.__class__.__name__ == 'GoogleNet':
        return ['pool5']
    elif model.__class__.__name__ == 'ResNet':
        return ['avgpool'] #layer4
    elif model.__class__.__name__ == 'Inception3':
        return ['Mixed_7c', 'branch_pool'] # ['conv2d_94'], 'mixed10'
    else: #TODO: guess layer for other networks?
        return None

def get_saliency(model, raw_input, input, label, device=None, method='integrate_grad', layer_path=None):
    if device == None or type(device) != torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    input = input.to(device)
    if label is not None:
        label = label.to(device)

    if input.grad is not None:
        input.grad.zero_()
    if label is not None and label.grad is not None:
        label.grad.zero_()
    model.eval()
    model.zero_grad()

    layer_path = layer_path or _get_layer_path(model)
    exp = _get_explainer(method, model, layer_path)
    saliency = exp.explain(input, label, raw_input)

    if saliency is not None:
        saliency = saliency.abs().sum(dim=1)[0].squeeze()
        saliency -= saliency.min()
        saliency /= (saliency.max() + 1e-20)

        return saliency.detach().cpu().numpy()
    else:
        return None

def get_image_saliency_results(model, raw_image, input, label,
                               device=None,
                               methods=['lime_imagenet', 'gradcam', 'smooth_grad',
                                        'guided_backprop', 'deeplift', 'grad_x_input'], 
                               layer_path=None):
    results = []
    for method in methods:
        sal = get_saliency(model, raw_image, input, label, device=device, method=method)

        if sal is not None:
            results.append(ImageSaliencyResult(raw_image, sal, method))
    return results

def get_image_saliency_plot(image_saliency_results, cols:int=2, figsize:tuple=None):
    import matplotlib.pyplot as plt # delayed import due to matplotlib threading issue

    rows = math.ceil(len(image_saliency_results) / cols)
    figsize=figsize or (8, 3 * rows)
    figure = plt.figure(figsize=figsize)
    
    for i, r in enumerate(image_saliency_results):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([]) 
        ax.set_title(r.title, fontdict={'fontsize': 24}) #'fontweight': 'light'

        #upsampler = nn.Upsample(size=(raw_image.height, raw_image.width), mode='bilinear')
        saliency_upsampled = skimage.transform.resize(r.saliency, 
                                                      (r.raw_image.height, r.raw_image.width),
                                                      mode='reflect')

        image_utils.show_image(r.raw_image, img2=saliency_upsampled, 
                             alpha2=r.saliency_alpha, cmap2=r.saliency_cmap, ax=ax)
    return figure