# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed


class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling


# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
