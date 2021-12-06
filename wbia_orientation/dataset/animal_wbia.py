# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from skimage import transform as skimage_transform
import imageio


class AnimalWbiaDataset(Dataset):
    def __init__(self, image_paths, bboxes, target_imsize, transform):
        self.image_paths = image_paths
        self.bboxes = bboxes
        self.target_imsize = tuple(target_imsize)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imageio.imread(self.image_paths[idx])
        if image is None:
            raise ValueError('Fail to read {}'.format(self.image_paths[id]))

        # Crop bounding box area
        x1, y1, w, h = self.bboxes[idx]
        image = image[y1 : y1 + h, x1 : x1 + w]
        if min(image.shape) < 1:
            # Use original image
            image = imageio.imread(self.image_paths[idx])
            self.bboxes[idx] = [0, 0, image.shape[1], image.shape[0]]

        # Resize image
        image = skimage_transform.resize(
            image, self.target_imsize, order=3, anti_aliasing=True
        )

        if self.transform is not None:
            image = self.transform(image)
        return image
