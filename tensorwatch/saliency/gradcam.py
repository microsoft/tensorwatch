import torch
from .backprop import VanillaGradExplainer

def _get_layer(model, key_list):
    if key_list is None:
        return None

    a = model
    for key in key_list:
        a = a._modules[key]

    return a

class GradCAMExplainer(VanillaGradExplainer):
    def __init__(self, model, target_layer_name_keys=None, use_inp=False):
        super(GradCAMExplainer, self).__init__(model)
        self.target_layer = _get_layer(model, target_layer_name_keys)
        self.use_inp = use_inp
        self.intermediate_act = []
        self.intermediate_grad = []
        self.handle_forward_hook = None
        self.handle_backward_hook = None
        self._register_forward_backward_hook()

    def _register_forward_backward_hook(self):
        def forward_hook_input(m, i, o):
            self.intermediate_act.append(i[0].data.clone())
        
        def forward_hook_output(m, i, o):
            self.intermediate_act.append(o.data.clone())
            
        def backward_hook(m, grad_i, grad_o):
            self.intermediate_grad.append(grad_o[0].data.clone())

        if self.target_layer is not None:
            if self.use_inp:
                self.handle_forward_hook = self.target_layer.register_forward_hook(forward_hook_input)
            else:
                self.handle_forward_hook = self.target_layer.register_forward_hook(forward_hook_output)
            
            self.handle_backward_hook = self.target_layer.register_backward_hook(backward_hook)

    def _reset_intermediate_lists(self):
        self.intermediate_act = []
        self.intermediate_grad = []

    def explain(self, inp, ind=None, raw_inp=None):
        self._reset_intermediate_lists()
        _ = super(GradCAMExplainer, self)._backprop(inp, ind)
        self.handle_forward_hook.remove()
        self.handle_backward_hook.remove()
        if len(self.intermediate_grad):
            grad = self.intermediate_grad[0]
            act = self.intermediate_act[0]
            
            weights = grad.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
            cam = weights * act
            cam = cam.sum(1).unsqueeze(1)

            cam = torch.clamp(cam, min=0)
            return cam
        else:
            return None

