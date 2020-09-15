# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path
import pickle
import csv

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def load_partial_weights(model, model_path, pretrained_state=None, cuda_avail=True):
    if pretrained_state is None:
        if cuda_avail:
            pretrained_state = torch.load(model_path)
        else:
            pretrained_state = torch.load(model_path, map_location=torch.device('cpu'))
            
    model_state = model.state_dict()
    #print(model_state.keys())
    transfer_state = {k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size()}
    #print('Loading weights for layers:', transfer_state.keys())
    not_in_model_state = [k for k,v in pretrained_state.items() if k not in model_state or v.size() != model_state[k].size()]
    print('Not loaded weights:', not_in_model_state)
    model_state.update(transfer_state)
    print(model.load_state_dict(model_state))  
    no_init = [k for k,v in model_state.items() if ('num_batches_tracked' not in k) and (k not in pretrained_state or v.size() != pretrained_state[k].size())]
    print('Randomly initialised weights', no_init)
    return transfer_state.keys(), not_in_model_state, no_init

def create_logger(cfg, cfg_path, phase='train', create_tb=True):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    cfg_name = os.path.split(os.path.split(cfg_path)[0])[-1]
    cfg_name += '_' + os.path.basename(cfg_path).split('.')[0]

    final_output_dir = root_output_dir / (cfg_name+'_'+cfg.VERSION)

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    #create dir for debug images
    debug_imgs_dir = os.path.join(final_output_dir, 'debug_images')
    if not os.path.exists(debug_imgs_dir): os.makedirs(debug_imgs_dir)
    
    if create_tb:
        tensorboard_log_dir = Path(cfg.LOG_DIR) / \
            (cfg_name + '_' + cfg.VERSION + ' ' + time_str)
    
        print('=> creating {}'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    
        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return logger, str(final_output_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

def num_trainable_params(model, print_list = False):  
    """Count number of trainable parameters
    """
    n_trainable = 0
    n_total = 0
    #for child in model.children():
    for param in model.parameters():
        n_total += param.nelement()
        if param.requires_grad == True:
            n_trainable += param.nelement()
    print('Trainable {:,} parameters out of {:,}'.format(n_trainable, n_total))
    if print_list:
        print('Trainable parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t {} \t {} \t {:,}'.format(name, param.size(), param.numel()))
    return n_trainable

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

def save_object(obj, filename):
    """Save python object to a file"""
    folder = os.path.split(filename)[0]
    if not os.path.exists(folder): os.makedirs(folder)
    print('Saving the data to {}'.format(filename))
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
        
def unnormalize(batch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], use_gpu=False):
    """Reverse normalization applied to image by transformations
    """
    B = batch_image.shape[0]
    H = batch_image.shape[2]
    W = batch_image.shape[3]
    t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3,H,W).contiguous().view(1,3,H,W)
    t_std = torch.FloatTensor(std).view(3,1,1).expand(3,H,W).contiguous().view(1,3,H,W)
    if use_gpu:
        t_mean = t_mean.cuda()
        t_std = t_std.cuda()
    batch_image_unnorm = batch_image * t_std.expand(B,3,H,W) +t_mean.expand(B,3,H,W)
    return batch_image_unnorm

def save_res_csv(results, filename):
    """Save dictionary with results to a csv file
    Input:
    results: dictionary, keys will be headers for the csv file, values - rows
    filename: string, name for csv file (eg. results.csv)
    """
    exp_header = [k for k, v in results.items()]
    exp_data = [v for k, v in results.items()]

    #Log iteration results. If file does not exist yet, create file with header
    if not os.path.isfile(filename):
        with open(filename, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(exp_header)
            print('File {} is created'.format(filename))
    #TODO add check if file exists, that header row is the same as header from data

    with open(filename, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(exp_data)
        
class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
