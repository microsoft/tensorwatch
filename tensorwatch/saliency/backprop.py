import numpy as np
from torch.autograd import Variable, Function
import torch
import types


class VanillaGradExplainer(object):
    def __init__(self, model):
        self.model = model

    def _backprop(self, inp, ind):
        inp.requires_grad = True
        if inp.grad is not None:
            inp.grad.zero_()
        if ind.grad is not None:
            ind.grad.zero_()
        self.model.eval()
        self.model.zero_grad()

        output = self.model(inp)
        if ind is None:
            ind = output.max(1)[1]
        grad_out = output.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        output.backward(grad_out)
        return inp.grad

    def explain(self, inp, ind=None, raw_inp=None): 
        return self._backprop(inp, ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GradxInputExplainer, self).__init__(model)

    def explain(self, inp, ind=None, raw_inp=None):
        grad = self._backprop(inp, ind)
        return inp * grad


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(SaliencyExplainer, self).__init__(model)

    def explain(self, inp, ind=None, raw_inp=None):
        grad = self._backprop(inp, ind)
        return grad.abs()


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100):
        super(IntegrateGradExplainer, self).__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None, raw_inp=None):
        grad = 0
        inp_data = inp.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / self.steps


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(DeconvExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


class GuidedBackpropExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GuidedBackpropExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float()
                mask2 = (grad_output > 0).float()
                grad_inp = mask1 * mask2 * grad_output
                grad_output.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


# modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
class SmoothGradExplainer(object):
    def __init__(self, model, base_explainer=None, stdev_spread=0.15,
                nsamples=25, magnitude=True):
        self.base_explainer = base_explainer or VanillaGradExplainer(model)
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude

    def explain(self, inp, ind=None, raw_inp=None):
        stdev = self.stdev_spread * (inp.max() - inp.min())

        total_gradients = 0

        for i in range(self.nsamples):
            noise = torch.randn_like(inp) * stdev

            noisy_inp = inp + noise
            noisy_inp.retain_grad()
            grad = self.base_explainer.explain(noisy_inp, ind)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples