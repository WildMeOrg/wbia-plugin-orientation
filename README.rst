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
