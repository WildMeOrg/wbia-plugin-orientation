# Orientation network

## Overview
TODO


## Results
Accuracy of predicting an angle of orientation on **a  test set**. Accuracy is computed for **10 and 15 degrees** thresholds:
| Dataset              | Acc@10   | Acc@15  |
| -------------        |:--------:| :------:|
| Seadragon heads      | 95.45%   | 97.60%  |
| Seaturtle heads      | 82.42%   | 91.81%  |
| Spotted Dolphin      | 80.02%   | 89.22%  |
| Manta Ray            | 66.67%   |  73.90% |


## Examples
TODO


## Implementation details
### Dependencies

* Python >= 3.7
* PyTorch >= 1.5

### Data
Data used for training and evaluation:
* sea turtle head parts - [orientation.seaturtle.coco.tar.gz](https://cthulhu.dyn.wildme.io/public/datasets/orientation.seaturtle.coco.tar.gz)
* sea dragon head parts - [orientation.seadragon.coco.tar.gz](https://cthulhu.dyn.wildme.io/public/datasets/orientation.seadragon.coco.tar.gz)
* manta ray body annotations - [orientation.mantaray.coco.tar.gz](https://cthulhu.dyn.wildme.io/public/datasets/orientation.mantaray.coco.tar.gz)
* spotted dolphin body annotations - [orientation.spotteddolphin.coco.tar.gz](https://cthulhu.dyn.wildme.io/public/datasets/orientation.spotteddolphin.coco.tar.gz)
* hammerhead shark body annotations - [orientation.hammerhead.coco.tar.gz](https://cthulhu.dyn.wildme.io/public/datasets/orientation.hammerhead.coco.tar.gz)
* right whale bonnet parts - [orientation.rightwhale.coco.tar.gz](https://cthulhu.dyn.wildme.io/public/datasets/orientation.rightwhale.coco.tar.gz)
### Data augmentations
#### Preprocessing
Each dataset is preprocessed to speed-up image loading during training. At the first time of running a training or a testing script on a dataset the following operations are applied:
* an object is cropped based on a segmentation boudnding box from annotations with a padding around equal to the half size of the box to allow for image augmentations
* an image is resized so the smaller side is equal to the double size of a model input; the aspect ratio is preserved.

The preprocessed dataset is saved in `data` directory.

#### Augmentations
During the training the data is augmented online in the following way:
* Random Horizontal Flips
* Random Vertical Flips
* Random Rotations
* Random Scale
* Random Crop
* Color Jitter (variations in brightness, hue, constrast and saturation)

Both training and testing data are resized to the model input size and normalized.

### Training
Run the training script:
```
python train.py --cfg <path_to_config_file> <additional_optional_params>
```
Configuration files are listed in `experiments` folder. For example, the following line trains the model with parameters specified in the config file:
```
python train.py --cfg experiments/3_hrnet_coords.yaml
```
Parameters from the config can be overwritten with command line parameters. To train on a different dataset with the same configuration, provide a dataset name as a parameter:
```
python train.py --cfg experiments/3_hrnet_coords.yaml DATASET.NAME spotteddolphin
```
### Testing
The test script evaluates on the test set with the best model saved during training:
```
python test.py --cfg <path_to_config_file> <additional_optional_params>
```
For example:
```
python test.py --cfg experiments/3_hrnet_coords.yaml DATASET.TEST_SET test2020
```
Averaging results over flips during test time yeilds slightly better accuracy:
```
python test.py --cfg experiments/3_hrnet_coords.yaml DATASET.TEST_SET test2020 TEST.HFLIP True TEST.VFLIP True
```
