import time
from collections import OrderedDict
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn

from .compute_madd import compute_madd
from .compute_flops import compute_flops
from .compute_memory import compute_memory


class ModelHook(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = torch.rand(*self._input_size)  # add module duration time
        x = x.to(next(model.parameters()).device)
        self._model.eval()
        self._model(x)

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        if hasattr(module, 'parameter_quantity'):
            return

        # register variables for each module to hold values we will compute
        module.register_buffer('parameter_quantity', torch.zeros(1).int())
        module.register_buffer('inference_memory', torch.zeros(1).long())
        module.register_buffer('input_shape', torch.zeros(3).int())
        module.register_buffer('output_shape', torch.zeros(3).int())
        module.register_buffer('MAdd', torch.zeros(1).long())
        module.register_buffer('duration', torch.zeros(1).float())
        module.register_buffer('Flops', torch.zeros(1).long())
        module.register_buffer('Memory', torch.zeros(2).long())

    def _to_seq(self, x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, Sequence):
            res = []
            for xi in x:
                res += self._to_seq(xi)
            return res
        return []

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            inputs, outputs = self._to_seq(input), self._to_seq(output)
            module.input_shape = torch.from_numpy(
                np.array(inputs[0].size(), dtype=np.int32))
            module.output_shape = torch.from_numpy(
                np.array(outputs[0].size(), dtype=np.int32))

            parameter_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameter_quantity += (0 if p is None else torch.numel(p.data))
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.long))

            inference_memory = 1
            for oi in outputs:
                for s in oi.size():
                    inference_memory *= s
            # memory += parameters_number  # exclude parameter memory
            inference_memory = inference_memory * 4 / (1024 ** 2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            madd = compute_madd(module, inputs, outputs)
            flops = compute_flops(module, inputs, outputs)
            Memory = compute_memory(module, inputs, outputs)

            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            module.Flops = torch.from_numpy(
                np.array([flops], dtype=np.int64))
            Memory = np.array(Memory, dtype=np.int32) * \
                     sum(oi.cpu().detach().numpy().itemsize for oi in outputs)
            module.Memory = torch.from_numpy(Memory)

            return output

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = []
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                leaf_modules.append((name, m))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))
