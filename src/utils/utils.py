# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
import os
import logging
import time
import pickle
import torch
import torch.optim as optim


def create_logger(cfg, cfg_path, phase='train', create_tb=True):
    root_output_dir = cfg.OUTPUT_DIR
    # set up logger
    if not os.path.exists(root_output_dir):
        print('=> creating {}'.format(root_output_dir))
        os.makedirs(root_output_dir)

    cfg_name = os.path.split(cfg_path)[-1].split('.')[0]

    dataset_name = cfg.DATASET.NAME

    final_output_dir = os.path.join(root_output_dir,
                                    dataset_name+'_'+cfg_name+'_'+cfg.VERSION)

    if not os.path.exists(final_output_dir):
        print('=> creating {}'.format(final_output_dir))
        os.makedirs(final_output_dir)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # Create dir for debug images
    debug_imgs_dir = os.path.join(final_output_dir, 'debug_images')
    if not os.path.exists(debug_imgs_dir):
        os.makedirs(debug_imgs_dir)

    if create_tb:
        tensorboard_log_dir = os.path.join(cfg.LOG_DIR,
                                           dataset_name + '_' +
                                           cfg_name + '_' +
                                           cfg.VERSION + ' ' +
                                           time_str)

        if not os.path.exists(tensorboard_log_dir):
            print('=> creating {}'.format(tensorboard_log_dir))
            os.makedirs(tensorboard_log_dir)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return logger, str(final_output_dir)


def get_optimizer(cfg, model):
    """ Create simple optimizer for model based on config """
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


def save_object(obj, filename):
    """Save python object to a file using pickle"""
    folder = os.path.split(filename)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('Saving the data to {}'.format(filename))
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def unnormalize(batch_image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                use_gpu=False):
    """Reverse normalization applied to batch of images """
    B = batch_image.shape[0]
    H = batch_image.shape[2]
    W = batch_image.shape[3]
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, H, W) \
                  .contiguous().view(1, 3, H, W)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, H, W) \
                 .contiguous().view(1, 3, H, W)
    if use_gpu:
        t_mean = t_mean.cuda()
        t_std = t_std.cuda()
    batch_image_unnorm = batch_image * t_std.expand(B, 3, H, W) + \
        t_mean.expand(B, 3, H, W)
    return batch_image_unnorm


def load_partial_weights(model, model_path, pretrained_state=None,
                         cuda_avail=True):
    """ Load partial weights for model """
    if pretrained_state is None:
        if cuda_avail:
            pretrained_state = torch.load(model_path)
        else:
            pretrained_state = torch.load(model_path,
                                          map_location=torch.device('cpu'))

    model_state = model.state_dict()
    transfer_state = {k: v for k, v in pretrained_state.items()
                      if k in model_state
                      and v.size() == model_state[k].size()}

    not_in_model_state = [k for k, v in pretrained_state.items()
                          if k not in model_state
                          or v.size() != model_state[k].size()]

    print('Not loaded weights:', not_in_model_state)
    model_state.update(transfer_state)

    print(model.load_state_dict(model_state))
    no_init = [k for k, v in model_state.items()
               if ('num_batches_tracked' not in k)
               and (k not in pretrained_state
               or v.size() != pretrained_state[k].size())]

    print('Randomly initialised weights', no_init)
    return transfer_state.keys(), not_in_model_state, no_init


class AverageMeterSet:
    """Computes and stores average and current values for a set of meters"""
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val
                for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg
                for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum
                for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count
                for name, meter in self.meters.items()}


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
