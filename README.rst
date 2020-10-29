=====================
Wildbook IA - wbia_id
=====================

ID Plug-in Example - Part of the WildMe / Wildbook IA Project.

An example of how to design and use a Python module as a plug-in in the WBIA system

Installation
------------

.. code:: bash

    ./run_developer_setup.sh

REST API
--------

With the plugin installed, register the module name with the `WBIAControl.py` file
in the wbia repository located at `wbia/wbia/control/WBIAControl.py`.  Register
the module by adding the string (for example, `wbia_plugin_identification_example`) to the
list `AUTOLOAD_PLUGIN_MODNAMES`.

Then, load the web-based WBIA IA service and open the URL that is registered with
the `@register_api decorator`.

.. code:: bash

    cd ~/code/wbia/
    python dev.py --web

Navigate in a browser to http://127.0.0.1:5000/api/plugin/example/helloworld/ where
this returns a formatted JSON response, including the serialized returned value
from the `wbia_plugin_identification_example_hello_world()` function

.. code:: text

    {"status": {"cache": -1, "message": "", "code": 200, "success": true}, "response": "[wbia_plugin_identification_example] hello world with WBIA controller <WBIAController(testdb1) at 0x11e776e90>"}

Python API
----------

.. code:: bash

    python

    Python 2.7.14 (default, Sep 27 2017, 12:15:00)
    [GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.37)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import wbia
    >>> ibs = wbia.opendb()

    [ibs.__init__] new IBEISController
    [ibs._init_dirs] ibs.dbdir = u'/Datasets/testdb1'
    [depc] Initialize ANNOTATIONS depcache in u'/Datasets/testdb1/_ibsdb/_wbia_cache'
    [depc] Initialize IMAGES depcache in u'/Datasets/testdb1/_ibsdb/_wbia_cache'
    [ibs.__init__] END new WBIAController

    >>> ibs.wbia_plugin_identification_example_hello_world()
    '[wbia_plugin_identification_example] hello world with WBIA controller <WBIAController(testdb1) at 0x10b24c9d0>'

The function from the plugin is automatically added as a method to the ibs object
as `ibs.wbia_plugin_identification_example_hello_world()`, which is registered using the
`@register_ibs_method decorator`.

Code Style and Development Guidelines
-------------------------------------

Contributing
~~~~~~~~~~~~

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any commit you make. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code:: bash

    git add .
    pre-commit run

    # or

    pre-commit run --all-files

Brunette
~~~~~~~~

Our code base has been formatted by Brunette, which is a fork and more configurable version of Black (https://black.readthedocs.io/en/stable/).

Flake8
~~~~~~

Try to conform to PEP8.  You should set up your preferred editor to use flake8 as its Python linter, but pre-commit will ensure compliance before a git commit is completed.

To run flake8 from the command line use:

.. code:: bash

    flake8


This will use the flake8 configuration within ``setup.cfg``,
which ignores several errors and stylistic considerations.
See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.

PyTest
~~~~~~

Our code uses Google-style documentation tests (doctests) that uses pytest and xdoctest to enable full support.  To run the tests from the command line use:

.. code:: bash

    pytest

To run doctests with `+REQUIRES(--web-tests)` do:

.. code:: bash

    pytest --web-tests

# Orientation network

## Overview
TODO


## Results
Accuracy of predicting an angle of orientation on **a test set**. Accuracy is computed for **10 and 15 degrees** thresholds:
| Dataset              | Acc@10   | Acc@15  |
| -------------        |:--------:| :------:|
| Seadragon heads      | 95.45%   | 97.60%  |
| Seaturtle heads      | 82.42%   | 91.81%  |
| Spotted Dolphin      | 80.02%   | 89.22%  |
| Manta Ray            | 66.67%   | 73.90% |


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
