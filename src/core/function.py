# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
 
import time
import logging
import torch

from core.evaluate import evaluate_orientaion
from utils.vis import save_debug_images
from utils.utils import AverageMeterSet

torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


def train(config, train_loader, model, loss_func, optimizer, epoch, output_dir, writer_dict):

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
        
    #Iterate over data loader
    for i, (images, xc, yc, xt, yt, w, theta) in enumerate(train_loader):
        end = time.time()
        bs = images.size(0)
        target_output = torch.stack([xc, yc, xt, yt, w], dim=1)
        
        if config.USE_GPU:
            images = images.cuda(non_blocking=True)
            target_output = target_output.cuda(non_blocking=True)
                        
        # Compute output of Orientation Network
        output = model(images)
        
        #Compute loss and backpropagate
        loss = loss_func(output, target_output)
        meters.update('train_loss', loss.item(), bs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Compute performance: accuracy of theta and coords
        perf = evaluate_orientaion(output.detach().cpu().numpy(), 
                                         target_output.detach().cpu().numpy(),
                                         theta,
                                         theta_thr = config.TEST.THETA_THR, 
                                         theta_source = 'annot')
        
        meters.update('train_err_theta', perf['err_theta'], bs)
        meters.update('train_acc_theta', perf['acc_theta'], bs)
        meters.update('train_err_xcyc',  perf['err_xcyc'], bs)
        meters.update('train_err_xtyt',  perf['err_xtyt'], bs)
        meters.update('train_err_w', perf['err_w'], bs)
        
        # measure elapsed time
        batch_time = time.time() - end

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\tBatch time {batch_time:.3f}s\t'. \
                            format(epoch, i, len(train_loader), batch_time=batch_time)
            for key, val in meters.meters.items():
                msg += '{} {:.4f} ({:.4f})\t'.format(key, val.val, val.avg) 
            logger.info(msg)

            #Update tensorboard logs
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']

            for key, val in meters.values().items():
                writer.add_scalar(key, val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1           
            
            #Save some image with detected points
            save_debug_images(config, images.cpu(), 
                              target_output.cpu()*config.MODEL.IMAGE_SIZE[0], 
                              output.detach().cpu()*config.MODEL.IMAGE_SIZE[0], 
                              theta, None, 'train_{}'.format(i), output_dir)
            
        if config.LOCAL and i > 3:
            break


def validate(config, val_loader, val_dataset, model, loss_func, output_dir, writer_dict=None):
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        for i, (images, xc, yc, xt, yt, w, theta) in enumerate(val_loader):
            bs = images.size(0)
            target_output = torch.stack([xc, yc, xt, yt, w], dim=1)
            
            if config.USE_GPU:
                images = images.cuda(non_blocking=True)
                target_output = target_output.cuda(non_blocking=True)
                
            # Compute output of Orientation Network 
            output = model(images)
            
            #TODO add flips at test time
#            if config.TEST.FLIP_TEST:
#                input_flipped = input_images.flip(3)
#                outputs_flipped, _ = model['spen'](input_flipped)
#
#                if isinstance(outputs_flipped, list):
#                    output_flipped = outputs_flipped[-1]
#                else:
#                    output_flipped = outputs_flipped
#
#                output_flipped = flip_back(output_flipped.cpu().numpy(),
#                                           val_dataset.flip_pairs)
#                output_flipped = torch.from_numpy(output_flipped.copy())
#                if config.USE_GPU:
#                    output_flipped = output_flipped.cuda()
#                output = (output + output_flipped) * 0.5

            loss = loss_func(output, target_output)
            meters.update('valid_loss', loss.item(), bs)
                
            #Compute accuracy on all examples
            perf = evaluate_orientaion(output.detach().cpu().numpy(), 
                                             target_output.detach().cpu().numpy(),
                                             theta,
                                             theta_thr = config.TEST.THETA_THR, 
                                             theta_source = 'annot')
            
            meters.update('valid_err_theta', perf['err_theta'], bs)
            meters.update('valid_acc_theta', perf['acc_theta'], bs)
            meters.update('valid_err_xcyc',  perf['err_xcyc'], bs)
            meters.update('valid_err_xtyt',  perf['err_xtyt'], bs)
            meters.update('valid_err_w', perf['err_w'], bs)
            
            #TODO Export annotations

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t'.format(i, len(val_loader))
                for key, val in meters.meters.items():
                    msg += '{} {:.4f} ({:.4f})\t'.format(key, val.val, val.avg)     
                logger.info(msg)

            #Save some image with detected points
            save_debug_images(config, images.cpu(), 
                              target_output.cpu()*config.MODEL.IMAGE_SIZE[0], 
                              output.detach().cpu()*config.MODEL.IMAGE_SIZE[0], 
                              theta, None, 'valid_{}'.format(i), output_dir)
                                
            if config.LOCAL and i>3:
                break

        #Update tensorboard
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            for key, val in meters.values().items():
                writer.add_scalar(key, val, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1
            
    return perf['acc_theta']