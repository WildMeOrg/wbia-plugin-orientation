# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
# ------------------------------------------------------------------------------
import logging
import os
from pycocotools.coco import COCO
import copy
import imageio
from skimage import transform as skimage_transform
import json
from skimage import img_as_ubyte

from torch.utils.data import Dataset
from utils.data_manipulation import rotate_point_by_angle
from utils.data_manipulation import increase_bbox
from utils.data_manipulation import to_origin
from utils.data_manipulation import resize_coords

logger = logging.getLogger(__name__)


class AnimalDataset(Dataset):
    """
    Class for birds full body and body parts dataset.
    """
    def __init__(self, cfg, is_train, transform=None, crop=True, resize=True):
        self.cfg = cfg
        self.is_train = is_train
        self.transform = transform
        self.split = self.cfg.DATASET.TRAIN_SET if self.is_train else self.cfg.DATASET.TEST_SET

        self.crop = crop
        self.resize = resize

        # Check if preprocessed dataset exists
        if self._preproc_db_exists():
            # Load preproc dataset
            self.db = self._get_preproc_db()
        else:
            # Load COCO annots and preproc
            db_coco = self._get_coco_db()
            self.db = self._preproc_db(db_coco=db_coco,
                                       expand=2.,
                                       min_size=2*max(self.cfg.MODEL.IMSIZE))

        logger.info('=> load {} samples from {} / {} dataset'.
                    format(len(self.db), self.cfg.DATASET.NAME, self.split))

    def __len__(self,):
        return len(self.db)

    def _get_preproc_db(self):
        """ Load preprocessed database """
        with open(self.prep_annots) as file:
            db = json.load(file)
        return db

    def _preproc_db_exists(self):
        """ Check if preprocessed dataset exists """
        if len(self.cfg.DATASET.SUFFIX) == 0:
            self.prep_dir = os.path.join(self.cfg.DATA_DIR,
                                         self.cfg.DATASET.NAME)
        else:
            self.prep_dir = os.path.join(self.cfg.DATA_DIR,
                                         '{}_{}'.format(self.cfg.DATASET.NAME,
                                                        self.cfg.DATASET.SUFFIX
                                                        ))
        self.prep_images = os.path.join(self.prep_dir, 'images', self.split)
        self.prep_annots = os.path.join(self.prep_dir,
                                        'annots',
                                        '{}.json'.format(self.split))

        if os.path.exists(self.prep_images) and os.path.exists(self.prep_annots):
            return True
        else:
            if not os.path.exists(self.prep_images):
                os.makedirs(self.prep_images)
            if not os.path.exists(os.path.split(self.prep_annots)[0]):
                os.makedirs(os.path.split(self.prep_annots)[0])
            return False

    def _get_coco_annot_file(self):
        """ Get name of file with annotations """
        coc_ann_file = os.path.join(self.cfg.COCO_ANNOT_DIR,
                                    'orientation.{}.coco'.format(self.cfg.DATASET.NAME),
                                    'annotations',
                                    'instances_{}.json'.format(self.split))
        return coc_ann_file

    def _get_coco_db(self):
        """ Get database from COCO anntations """
        coc_ann_file = self._get_coco_annot_file()
        coco = COCO(coc_ann_file)
        image_set_index = coco.getImgIds()
        num_images = len(image_set_index)
        logger.info('=> Found {} images in {}'.format(num_images, coc_ann_file))
        gt_db = []
        for index in image_set_index:
            gt_db.extend(self._load_image_annots(coco, index))
        return gt_db

    def _preproc_db(self, db_coco, expand, min_size):
        """Preprocess images by cropping area twice the size of bounding box
        and resizing to a smaller size for faster augmentation and loading
        """
        logger.info('Preprocessing database...')
        prep_gt_db = []
        for i, db_rec in enumerate(db_coco):
            # Read image
            image = imageio.imread(db_rec['image_path'])
            if image is None:
                logger.error('=> fail to read {}'.format(db_rec['image_path']))
                raise ValueError('Fail to read {}'.format(db_rec['image_path']))

            aa_big_box = db_rec['aa_big_box']
            aa_bbox = db_rec['aa_bbox']

            if self.crop:
                # Get box around axis-aligned bounding box
                x1, y1, bw, bh = increase_bbox(aa_big_box,
                                               expand,
                                               image.shape[:2],
                                               type='xyhw')

                # Crop image and coordinates
                image_cropped = image[y1:y1+bh, x1:x1+bw]
                if min(image_cropped.shape) < 1:
                    print('Skipped image {} Cropped to zero size.'.
                          format(db_rec['image_path']))
                    continue
                else:
                    image = image_cropped

                # Shift coordinates to new origin
                aa_big_box = to_origin(aa_big_box, (x1, y1))
                aa_bbox = to_origin(aa_bbox, (x1, y1))

            if self.resize:
                # Compute output size
                if image.shape[0] <= image.shape[1]:
                    out_size = (min_size,
                                int(image.shape[1] * min_size / image.shape[0]))
                else:
                    out_size = (int(image.shape[0] * min_size / image.shape[1]),
                                min_size)

                # Resize coordinates
                aa_big_box = resize_coords(aa_big_box, image.shape[:2], out_size)
                aa_bbox = resize_coords(aa_bbox, image.shape[:2], out_size)

                # Resize image
                image = skimage_transform.resize(image,
                                                 out_size,
                                                 order=3,
                                                 anti_aliasing=True)

            # Save image to processed folder
            im_filename = os.path.basename(db_rec['image_path'])
            new_filename = os.path.join(self.prep_images,
                                        '{}_{}{}'.format(os.path.splitext(im_filename)[0],
                                                         db_rec['obj_id'],
                                                         os.path.splitext(im_filename)[1])
                                        )
            imageio.imwrite(new_filename, img_as_ubyte(image))

            prep_gt_db.append({'image_path': new_filename,
                                'aa_bbox': aa_bbox,
                                'theta': db_rec['theta'],
                                'aa_big_box': aa_big_box,
                                'category_id': db_rec['category_id'],
                                'obj_id': db_rec['obj_id']
                                })
        # Save as json
        with open(self.prep_annots, 'w', encoding='utf-8') as f:
            json.dump(prep_gt_db, f, ensure_ascii=False, indent=4)

        return prep_gt_db

    def _annot_sanity_check(self, obj, image_path):
        """ Check annotations for consistency """
        consistency_flag = True
        aa_bbox = obj['bbox']
        aa_big_box = obj['segmentation_bbox']

        # Width and height of bounding box cannot be less or equal to 0
        if aa_bbox[2] <= 0 or aa_bbox[3] <= 0:
            consistency_flag = False

        # Width and height of bounding box cannot be less or equal to 0
        if aa_big_box[2] <= 0 or aa_big_box[3] <= 0:
            consistency_flag = False

        if not consistency_flag:
            logger.info('Skipping image {}'.format(image_path))
            logger.info('Check bounding box annotations: {}, {}'.
                        format(aa_bbox, aa_big_box))

        return consistency_flag

    def _select_annot(self, obj_cat):
        """ Select annotation to add to the dataset.
        The annotation is included:
            a. there is no list of selected categories in config
            b. object category in the list of selected categories in config"""
        if len(self.cfg.DATASET.SELECT_CATS_LIST) == 0 or \
           (len(self.cfg.DATASET.SELECT_CATS_LIST) > 0 and
           obj_cat in self.cfg.DATASET.SELECT_CATS_LIST):
            return True
        else:
            return False

    def _load_image_annots(self, coco, index):
        """ Get COCO annotations for an image by index """
        im_anns = coco.imgToAnns[index]
        image_path = self._get_image_path(coco.imgs[index]['file_name'])

        rec = []
        for i, obj in enumerate(im_anns):

            # Skip annotations that do not pass sanity check
            if not self._annot_sanity_check(obj, image_path):
                continue

            if self._select_annot(obj['category_id']):
                rec.append({
                    'image_path': image_path,
                    'aa_bbox': obj['bbox'],
                    'theta': obj['theta'],
                    'aa_big_box': obj['segmentation_bbox'],
                    'category_id': obj['category_id'],
                    'obj_id': i
                })
        return rec

    def _get_image_path(self, filename):
        """ Get full path to image in COCO annotations by image filename """
        image_path = os.path.join(self.cfg.COCO_ANNOT_DIR,
                                  'orientation.{}.coco'.format(self.cfg.DATASET.NAME),
                                  'images',
                                  self.split,
                                  filename)
        return image_path

    def __getitem__(self, idx):
        """ Get record from database and return sample for training"""
        db_rec = copy.deepcopy(self.db[idx])

        # A. Load original image
        image = imageio.imread(db_rec['image_path'])
        if image is None:
            logger.error('=> fail to read {}'.format(db_rec['image_path']))
            raise ValueError('Fail to read {}'.format(db_rec['image_path']))

        # B. Get center point (xc,yc), orientation point (xt, yt),
        # width (w) and theta (rotation angle)
        bbox_x, bbox_y, bbox_w, bbox_h = db_rec['aa_bbox']

        xc, yc = bbox_x + bbox_w/2, bbox_y + bbox_h/2
        theta = db_rec['theta']
        xt, yt = rotate_point_by_angle([xc, yc], [bbox_x + bbox_w/2, bbox_y],
                                       theta)
        w = bbox_w / 2

        # C. Transform image and corresponding parameters
        if self.transform:
            image, xc, yc, xt, yt, w, theta = self.transform((image, xc, yc, xt, yt, w, theta))

        return image, xc, yc, xt, yt, w, theta
