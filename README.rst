===============================
Wildbook IA - wbia_orientation
===============================

Orientation Plug-in - Part of the WildMe / Wildbook IA Project.

A plugin for automatic detection of object-oriented bounding box based on axis-aligned box.

Installation
------------

.. code:: bash

    ./run_developer_setup.sh

REST API
--------

With the plugin installed, register the module name with the `WBIAControl.py` file
in the wbia repository located at `wbia/wbia/control/WBIAControl.py`.  Register
the module by adding the string `wbia_plugin_orientation` to the
list `AUTOLOAD_PLUGIN_MODNAMES`.

Then, load the web-based WBIA IA service and open the URL that is registered with
the `@register_api decorator`.

.. code:: bash

    cd ~/code/wbia/
    python dev.py --web

.. TODO update Rest API
.. Navigate in a browser to http://127.0.0.1:5000/api/plugin/example/helloworld/ where this returns a formatted JSON response, including the serialized returned valuefrom the `wbia_plugin_identification_example_hello_world()` function

.. code:: text

..     {"status": {"cache": -1, "message": "", "code": 200, "success": true}, "response": "[wbia_plugin_identification_example] hello world with WBIA controller <WBIAController(testdb1) at 0x11e776e90>"}

Python API
----------

.. code:: bash

    python
    >>> import wbia
    >>> ibs = wbia.opendb()
    >>> species = 'spotteddolphin'
    >>> ibs = wbia_orientation._plugin.wbia_orientation_test_ibs(species)
    >>> aid_list = ibs.get_valid_aids()
    >>> aid_list = aid_list[:3]
    >>> output = ibs.wbia_plugin_detect_oriented_box(aid_list, species, False, False)

The function from the plugin is automatically added as a method to the ibs object
as `ibs.wbia_plugin_detect_oriented_box()`, which is registered using the
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

Results and Examples
---------------------

Quantitative and qualitative results are `here </wbia_orientation>`_


Implementation details
----------------------
Dependencies
~~~~~~~~~~~~
* Python >= 3.7
* PyTorch >= 1.5

Data
~~~~~~~~~~~~
Data used for training and evaluation:
 * sea turtle head parts - `orientation.seaturtle.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.seaturtle.coco.tar.gz>`_
 * sea dragon head parts - `orientation.seadragon.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.seadragon.coco.tar.gz>`_
 * manta ray body annotations - `orientation.mantaray.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.mantaray.coco.tar.gz>`_
 * spotted dolphin body annotations - `orientation.spotteddolphin.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.spotteddolphin.coco.tar.gz>`_
 * hammerhead shark body annotations - `orientation.hammerhead.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.hammerhead.coco.tar.gz>`_
 * right whale bonnet parts - `orientation.rightwhale.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.rightwhale.coco.tar.gz>`_
 * whale  shark - `orientation.whaleshark.coco.tar.gz <https://cthulhu.dyn.wildme.io/public/datasets/orientation.whaleshark.coco.tar.gz>`_

Data preprocessing
~~~~~~~~~~~~~~~~~~
Each dataset is preprocessed to speed-up image loading during training. At the first time of running a training or a testing script on a dataset the following operations are applied:
 * an object is cropped based on a segmentation boudnding box from annotations with a padding around equal to the half size of the box to allow for image augmentations
 * an image is resized so the smaller side is equal to the double size of a model input; the aspect ratio is preserved.

The preprocessed dataset is saved in `data` directory.

Data augmentations
~~~~~~~~~~~~~~~~~~
During the training the data is augmented online in the following way:
 * Random Horizontal Flips
 * Random Vertical Flips
 * Random Rotations
 * Random Scale
 * Random Crop
 * Color Jitter (variations in brightness, hue, contrast and saturation)

Both training and testing data are resized to the model input size and normalized.

Training
~~~~~~~~~~~~
Run the training script:

.. code:: bash

  python wbia_orientation/train.py --cfg <path_to_config_file> <additional_optional_params>

Configuration files are listed in `experiments` folder. For example, the following line trains the model with parameters specified in the config file:

.. code:: bash

  python wbia_orientation/train.py --cfg wbia_orientation/config/mantaray.yaml


Testing
~~~~~~~~~~~~
The test script evaluates on the test set with the best model saved during training:

.. code:: bash

  python wbia_orientation/test.py --cfg <path_to_config_file> <additional_optional_params>

For example:

.. code:: bash

  python wbia_orientation/test.py --cfg wbia_orientation/config/mantaray.yaml

By default, the accuracy of detected rotation angle is computed for a threshold of 10 degrees.
Pass a different value as a command line parameter to evaluate with another threshold:

.. code:: bash

  python wbia_orientation/test.py --cfg wbia_orientation/config/mantaray.yaml TEST.THETA_THR 15.
