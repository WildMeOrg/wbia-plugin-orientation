# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
import torch
import logging
import os
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy
import imageio

import numpy as np

from torch.utils.data import Dataset
from utils.data_manipulation import midpoint, rotate_point_by_angle

logger = logging.getLogger(__name__)


class AnimalDataset(Dataset):
    """
    Class for birds full body and body parts dataset.
    """
    def __init__(self, cfg, is_train, transform=None):
        self.cfg = cfg
        self.is_train = is_train
        self.transform = transform
        
        self.image_width = self.cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = self.cfg.MODEL.IMAGE_SIZE[1]
        
        coc_ann_file = self._get_annot_file()
        self.coco = COCO(coc_ann_file)
            
        # deal with class names
        #cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        #self.classes = cats
        #logger.info('=> classes: {}'.format(self.classes))
        #self.num_classes = len(self.classes)
        #self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        #self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        #self._coco_ind_to_class_ind = dict(
        #    [
        #        (self._class_to_coco_ind[cls], self._class_to_ind[cls])
        #        for cls in self.classes[1:]
        #    ]
        #)

        # load image file names
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)
        logger.info('=> Found {} images in {}'.format(self.num_images, coc_ann_file))

        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))
        
    def __len__(self,):
        return len(self.db)
        
    def _get_annot_file(self):
        """ Get name of file with annotations """
        split = self.cfg.DATASET.TRAIN_SET if self.is_train else self.cfg.DATASET.TEST_SET
        coc_ann_file = os.path.join(self.cfg.DATA_DIR, 
                                'orientation.{}.coco'.format(self.cfg.DATASET.NAME), 
                                'annotations', 
                                'instances_{}.json'.format(split))
        return coc_ann_file
    
    def _get_image_path_from_filename(self, filename):
        """ example: images / train2017 / 000000119993.jpg """
        split = self.cfg.DATASET.TRAIN_SET if self.is_train else self.cfg.DATASET.TEST_SET
        image_path = os.path.join(self.cfg.DATA_DIR, 
                                'orientation.{}.coco'.format(self.cfg.DATASET.NAME), 
                                'images', 
                                split,
                                filename)

        return image_path

    def _get_db(self):
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_orientation_annotation(index))
        return gt_db
    
    def _load_coco_orientation_annotation(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        
        [{'license': 3,
          'file_name': '000000000005.jpg',
          'coco_url': None,
          'height': 1800,
          'width': 2400,
          'date_captured': '2017-08-08 23:08:36',
          'flickr_url': None,
          'id': 5,
          'uuid': 'dbe7ac4a-fd28-43de-c4d8-328e1f8b33ef'}]

    
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # TOTO add sanity checks for annotations
        rec = []
        for obj in objs:
            rec.append({
                'image_path': self._get_image_path_from_filename(im_ann['file_name']),
                'axis_aligned_bbox': obj['bbox'],
                'theta': obj['theta'],
                'object_aligned_bbox': obj['segmentation'][0][:8],
                'axis_aligned_big_box': obj['segmentation_bbox'],
                'category_id': obj['category_id']
            })

        return rec
    
    def __getitem__(self, idx):
        """ Get record from database and return image with ground truth annotations
        """
        db_rec = copy.deepcopy(self.db[idx])
        
        #A. Load original image
        image = imageio.imread(db_rec['image_path'])
        if image is None:
            logger.error('=> fail to read {}'.format(db_rec['image_path']))
            raise ValueError('Fail to read {}'.format(db_rec['image_path']))
       
        #B. Get center point (xc,yc), orientation point (xt, yt), width (w) and theta (rotation angle)
        bbox_x, bbox_y, bbox_w, bbox_h = db_rec['axis_aligned_bbox']
        
        xc, yc = bbox_x + bbox_w/2, bbox_y + bbox_h/2
        theta = db_rec['theta']
        xt, yt = rotate_point_by_angle([xc, yc], [bbox_x + bbox_w/2, bbox_y], theta)
        w = bbox_w / 2
    
        #c. Transform image and corresponding parameters
        if self.transform:
            image, xc, yc, xt, yt, w, theta = self.transform((image, xc, yc, xt, yt, w, theta))
        
        return image, xc, yc, xt, yt, w, theta


    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        pass
