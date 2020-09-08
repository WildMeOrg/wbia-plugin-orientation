# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
import json

import numpy as np
import torch
from torch.autograd import Variable

from core.evaluate import accuracy, metrics_notvisible
from core.inference import get_final_preds, get_max_preds
from core.inference import augment_with_matrix
from utils.transforms import flip_back
from utils.vis import save_debug_images, save_reconstructed
from utils.utils import AverageMeterSet
from utils.utils import unnormalize

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)


def train(config, train_loader, model, loss_dict, optimizer, epoch,
          output_dir, writer_dict):
    """
    train_loader: target and target_weight are provided only for the subset of batch
                    
    model: dictionary of models
    loss_dict: dictionary of losses
    """
    meters = AverageMeterSet()
    if config.LOCAL and config.USE_GPU:
        print('\tAt the start of the epoch')
        print('Total memory {:,}'.format(torch.cuda.get_device_properties(0).total_memory))
        print('Memory cached {:,}'.format(torch.cuda.memory_cached(0)))
        print('Memory allocated {:,}'.format(torch.cuda.memory_allocated(0)))

    # switch to train mode
    for name in model:
        model[name].train()
        
    #Use semantic keypoints, e.g. left eye and right eye are the same
    #sem_kp = list(range(len(config.DATASET.LEGEND)))
    sem_kp = config.DATASET.SEMANTIC_KP_LABELS

    for i, sample in enumerate(train_loader):
        input_images, target, target_weight, _, meta, input_images_aug, target_aug, target_weight_aug, M, M_hm = sample
        end = time.time()
        #Get index when labelled examples start (unlabelled_idx, labelled_idx)
        bs = input_images.size(0)
        if config.DATASET.LABELLED_ONLY:
            lab_start = 0
        elif config.DATASET.UNLABELLED_ONLY:
            lab_start = bs
        else:
            lab_start = config.TRAIN.BS - config.TRAIN.BS_LABELLED
            assert lab_start < bs, 'Check number of labelled examples per batch, param cfg.TRAIN.BS_LABELLED'
            
        if config.USE_GPU:
            input_images = input_images.cuda(non_blocking=True)
            input_images_aug = input_images_aug.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            target_aug = target_aug.cuda(non_blocking=True)
            target_weight_aug = target_weight_aug.cuda(non_blocking=True)
            M = M.cuda(non_blocking=True)
            M_hm = M_hm.cuda(non_blocking=True)
            
        if config.LOCAL and config.USE_GPU:
            print('\tAfter variable allocation')
            print('Memory cached {:,}'.format(torch.cuda.memory_cached(0)))
            print('Memory allocated {:,}'.format(torch.cuda.memory_allocated(0)))
            
            
        # Compute output of Student Pose Estimation Network
        start_pen = time.time()
        hm, image_feat = model['spen'](input_images, collect_feat=config.MODEL.COLLECT_FEAT)
        if config.MODEL.TPEN:
            hm_aug, image_aug_feat = model['tpen'](input_images_aug, collect_feat=config.MODEL.COLLECT_FEAT)
            hm_aug = Variable(hm_aug.detach().data, requires_grad=False)
            image_aug_feat = Variable(image_aug_feat.detach().data, requires_grad=False)
        else:
            hm_aug, image_aug_feat = model['spen'](input_images_aug, collect_feat=config.MODEL.COLLECT_FEAT)
            
        if config.MODEL.KP_CLASS:
            #TODO use ground truth for labelled examples
            if lab_start < bs:
                hm_split = torch.split(torch.cat([hm[:lab_start], target[lab_start:]], dim=0), 1, 1)
            else:
                hm_split = torch.split(hm, 1, 1) #split heatmaps (bs, kp, h, w) into list  [(bs, 1, h, 2), ... ]
            #hm_cat = torch.reshape(hm.shape[0]*hm.shape[1], hm.shape[2], hm.shape[3])
            scores = [model['kp_class'](image_feat, hm_sp) for hm_sp in hm_split] # list [(bs, num_sem_kp)]
            scores = torch.cat(scores, 0)
            #scores = model['kp_class'](image_feat, hm_cat)
            
        if config.LOCAL and config.USE_GPU:
            print('\tAfter pen computations')
            print('Memory cached {:,}'.format(torch.cuda.memory_cached(0)))
            print('Memory allocated {:,}'.format(torch.cuda.memory_allocated(0)))
        
            
        if config.LOCAL:
            print('Pose estimation network time {:.3f}s'.format(time.time()-start_pen))
        
        # Compute output of Student Reconstruction Network
        if config.LOSS.RECONSTRUCTION or config.LOSS.RECONSTRUCTION_CONS:
            if config.TRAIN.USE_GT_HM_REC:
                #use ground truth heatmaps for labelled examples
                image_rec = model['srn'](image_aug_feat, torch.cat([hm[:lab_start], target[lab_start:]], dim=0))
                image_aug_rec = model['srn'](image_feat, torch.cat([hm_aug[:lab_start], target_aug[lab_start:]], dim=0))
            else:
                image_rec = model['srn'](image_aug_feat, hm)
                image_aug_rec = model['srn'](image_feat, hm_aug)
                
        
        sup_loss = 0.
        cons_loss = 0.
        rec_loss = 0.
        rec_cons_loss = 0.
        kp_class_loss = 0.
        kp_sim_loss = 0.
        
        # A. Compute supervised loss for labelled examples
        if 'supervised' in loss_dict:
            sup_loss_st = config.LOSS.SUPERVISED_WEIGHT * loss_dict['supervised'](hm[lab_start:], target[lab_start:], target_weight[lab_start:])
            sup_loss_t = config.LOSS.SUPERVISED_WEIGHT * loss_dict['supervised'](hm_aug[lab_start:], target_aug[lab_start:], target_weight_aug[lab_start:])
            sup_loss = sup_loss_st + sup_loss_t
            sup_loss /= 2
            if config.MODEL.TPEN:
                meters.update('train_loss_sup_st', sup_loss_st.item(), bs-lab_start)
                meters.update('train_loss_sup_t', sup_loss_t.item(), bs-lab_start)
            meters.update('train_loss_sup', sup_loss.item(), bs-lab_start)
        
        # B. Compute consistency loss for all examples
        if 'consistency' in loss_dict:
            #Augment heatmap of input images
            hm_aug_matrix = augment_with_matrix(M_hm, hm)
            cons_loss = config.LOSS.CONSISTENCY_WEIGHT * loss_dict['consistency'](hm_aug, hm_aug_matrix) #/ minibatch_size
            meters.update('train_loss_cons', cons_loss.item(), bs)          
            
        # C. Compute reconstruction loss for all examples
        if 'reconstruction' in loss_dict:
            #Compare with input images
            rec_loss += 0.5 * loss_dict['reconstruction'](image_rec, unnormalize(input_images, use_gpu=config.USE_GPU))
            rec_loss += 0.5 * loss_dict['reconstruction'](image_aug_rec, unnormalize(input_images_aug, use_gpu=config.USE_GPU))
            
            rec_loss *= config.LOSS.RECONSTRUCTION_WEIGHT
            meters.update('train_loss_rec', rec_loss.item(), bs) 
        
        #D. Compute reconstruction consistency for all examples
        if 'reconstruction_consistency' in loss_dict:
            #Invert augmentation
            image_rec_aug_matrix = augment_with_matrix(M, image_rec)
            #Compare reconstructed images
            rec_cons_loss = config.LOSS.RECONSTRUCTION_CONS_WEIGHT * loss_dict['reconstruction_consistency'](image_rec_aug_matrix, image_aug_rec)
            meters.update('train_loss_rec_cons', rec_cons_loss.item(), bs)  
            
        #E. Compute classification loss for labelled keypoint representation
        if 'kp_class' in loss_dict:
            gt_classes = torch.tensor(bs*sem_kp).view(bs, -1).transpose(1,0).reshape(bs*len(sem_kp),)
            #print('gt shape, scroes shape', gt_classes.shape, scores.shape)
            if config.USE_GPU:
                gt_classes = gt_classes.cuda(non_blocking=True)
            kp_class_loss = config.LOSS.KP_CLASS_WEIGHT * loss_dict['kp_class'](scores, gt_classes)
            meters.update('train_loss_kpclass', kp_class_loss.item(), gt_classes.size(0))
            
        # F. Keypoint  similarity loss
        if 'kp_similarity' in loss_dict:
            #TODO weights
            kp_sim_loss = config.LOSS.KP_SIMILARITY_WEIGHT * loss_dict['kp_similarity'](image_feat, hm)
            meters.update('train_loss_kpsim', kp_sim_loss.item(), bs)
            
        #Update weights
        loss = sup_loss + cons_loss + rec_loss + rec_cons_loss + kp_class_loss + kp_sim_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if config.MODEL.TPEN:
            update_teacher_weights(model['spen'], model['tpen'], 0.999, writer_dict['train_global_steps'])

        #Record loss
        meters.update('train_loss', loss.item(), bs)

        #Compute accuracy on all examples
        _, avg_acc, cnt, _ = accuracy(hm.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        meters.update('train_acc', avg_acc, cnt)
        
        if 'kp_class' in loss_dict:
            #Evaluate accuracy
            #gt_classes = torch.tensor(bs*sem_kp).view(bs, -1).transpose(1,0).reshape(bs*len(sem_kp),)
            _, predicted = torch.max(scores, 1)
            acc = (predicted.cpu() == gt_classes.cpu()).sum().item() / gt_classes.size(0)
            meters.update('train_acc_kp', acc, gt_classes.size(0))
            
        #Compute accuracy on labelled examples
        if lab_start < bs:
            _, avg_acc, cnt, _ = accuracy(hm[lab_start:].detach().cpu().numpy(),
                                         target[lab_start:].detach().cpu().numpy())
            meters.update('train_acc_lab', avg_acc, cnt)
        
        #Compute accuracy on unlabelled examples
        if lab_start > 0:
            _, avg_acc, cnt, _ = accuracy(hm[:lab_start].detach().cpu().numpy(),
                                         target[:lab_start].detach().cpu().numpy())
            meters.update('train_acc_unlab', avg_acc, cnt)
        
        pred = get_max_preds(hm.detach().cpu().numpy())[0] * 4   
        
        # measure elapsed time
        batch_time = time.time() - end

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t'\
                  'Batch time {batch_time:.3f}s\t'.format(epoch, i, len(train_loader), batch_time=batch_time)
                  
            for key, val in meters.meters.items():
                    msg += '{} {:.4f} ({:.4f})\t'.format(key, val.val, val.avg) 
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            #Add losses and accuracy to tensorboard
            for key, val in meters.values().items():
                writer.add_scalar(key, val, global_steps)
                
            writer_dict['train_global_steps'] = global_steps + 1           
            
            #Save some image with localized keypoints
            #Localization on labelled examples
            debug_imgs_dir = os.path.join(output_dir, 'debug_images')
            if not os.path.exists(debug_imgs_dir): os.makedirs(debug_imgs_dir)
                    
            if lab_start < bs:
                prefix_lab = '{}_{}'.format(os.path.join(debug_imgs_dir, 'train_lab'), i)
                save_debug_images(config, input_images[lab_start:], 
                                  meta['joints'][lab_start:], meta['joints_vis'][lab_start:], 
                                  target[lab_start:], pred[lab_start:], hm[lab_start:], prefix_lab)
            #Unlabelled examples
            if lab_start > 0:
                prefix_unlab = '{}_{}'.format(os.path.join(debug_imgs_dir, 'train_unlab'), i)
                
                save_debug_images(config, input_images[:lab_start], 
                                  meta['joints'][:lab_start], meta['joints_vis'][:lab_start], 
                                  target[:lab_start], pred[:lab_start], hm[:lab_start], prefix_unlab)
            
            if config.LOSS.RECONSTRUCTION or config.LOSS.RECONSTRUCTION_CONS:
                save_reconstructed(config, input_images, input_images_aug, image_rec, image_aug_rec, output_dir, global_steps)
            
        if config.LOCAL and i > 3:
            break


def validate(config, val_loader, val_dataset, model, loss_dict, output_dir, writer_dict=None):
    meters = AverageMeterSet()

    # switch to evaluate mode
    model['spen'].eval()
    sem_kp = config.DATASET.SEMANTIC_KP_LABELS

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    export_annots = []
    target_weights = []
    pred_max_vals_valid = []
    with torch.no_grad():
        #end = time.time()
        for i, (input_images, target, target_weight, meta) in enumerate(val_loader):
            bs = input_images.size(0)
            
            if config.USE_GPU:
                input_images = input_images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                
            # Compute output of Student Pose Estimation Network 
            output, image_feat = model['spen'](input_images, collect_feat=config.MODEL.COLLECT_FEAT)
            
            if config.MODEL.KP_CLASS:
                hm_split = torch.split(output, 1, 1) #split heatmaps (bs, kp, h, w) into list  [(bs, 1, h, 2), ... ]
                scores = [model['kp_class'](image_feat, hm_sp) for hm_sp in hm_split] # list [(bs, num_sem_kp)]
                scores = torch.cat(scores, 0)
                
            target_weights.append(target_weight)

            if config.TEST.FLIP_TEST:
                input_flipped = input_images.flip(3)
                outputs_flipped, _ = model['spen'](input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy())
                if config.USE_GPU:
                    output_flipped = output_flipped.cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            loss = 0.
            if 'supervised' in loss_dict:
                loss = loss_dict['supervised'](output, target, target_weight)
                meters.update('valid_loss_sup', loss, bs)
                
            if 'kp_class' in loss_dict:
                #Evaluate accuracy
                gt_classes = torch.tensor(bs*sem_kp).view(bs, -1).transpose(1,0).reshape(bs*len(sem_kp),)
                _, predicted = torch.max(scores, 1)
                acc = (predicted.cpu() == gt_classes).sum().item() / gt_classes.size(0)
                meters.update('valid_acc_kp', acc, gt_classes.size(0))
            
            # measure accuracy and record loss
            meters.update('valid_loss', loss.item(), bs)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            pred_maxvals = get_max_preds(output.cpu().numpy())[1]

            meters.update('valid_acc', avg_acc, cnt)
            pred_max_vals_valid.append(pred_maxvals)

            # measure elapsed time
            #batch_time = time.time() - end
            #end = time.time()
            
            if config.DATASET.CENTER_SCALE:
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)
            else:
                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), None, None)

            if config.DATASET.CENTER_SCALE:
                all_preds[idx:idx + bs, :, 0:2] = preds[:, :, 0:2] # * 4   # to go from hm size 64 to image size 256
            else:
                all_preds[idx:idx + bs, :, 0:2] = preds[:, :, 0:2] * 4 
            all_preds[idx:idx + bs, :, 2:3] = maxvals
            
            if config.DATASET.CENTER_SCALE:
            # double check this all_boxes parts
                all_boxes[idx:idx + bs, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + bs, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + bs, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + bs, 5] = score
            image_path.extend(meta['image'])
            
            #Export annotations
            for j in range(bs):
                annot = {"joints_vis": maxvals[j].squeeze().tolist(),
                         "joints": (pred[j]*4).tolist(),
                         "image": meta['image'][j]
                        }
                export_annots.append(annot)

            idx += bs

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t'.format(
                          i, len(val_loader))
                for key, val in meters.meters.items():
                    msg += '{} {:.4f} ({:.4f})\t'.format(key, val.val, val.avg)     
                
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'debug_images', 'val'), i)
                save_debug_images(config, input_images, meta['joints'], meta['joints_vis'], target, pred*4, output,
                                  prefix)
                                
            if config.LOCAL and i>3:
                break

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value_column(name_value, model_name)
        else:
            _print_name_value_column(name_values, model_name)
            
        #Compute and display accuracy, precision and recall
#        target_weights = torch.cat(target_weights, dim=0).squeeze()
#        gt_vis = ~target_weights.cpu().numpy().astype(bool)
#        pred_max_vals_valid = np.concatenate(pred_max_vals_valid, axis=0)
#        msg_notvis = metrics_notvisible(pred_max_vals_valid, gt_vis)
#        logger.info(msg_notvis)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            for key, val in meters.values().items():
                writer.add_scalar(key, val, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1
            
#    with open(os.path.join(output_dir, '{}_pred_annots_{}.json'.format(config.DATASET.TEST_SET, time.strftime('%Y-%m-%d-%H-%M'))), 'w', encoding='utf-8') as f:
#        json.dump(export_annots, f, ensure_ascii=False, indent=4)

    return perf_indicator

def update_teacher_weights(student_model, teacher_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data.mul_(alpha).add_(1 - alpha, s_param.data)

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )
    
# markdown format output
def _print_name_value_column(name_value, full_arch_name):
    logger.info('| Landmark | Accuracy |')
    for name, value in name_value.items():
        logger.info('| {} | {:.3f} |'.format(name, value))
