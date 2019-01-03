============
Contributing
============

Dev setup
=========

Install dev packages
--------------------

Specify the ``[dev]`` option to install the development packages:

.. code-block::  bash

    pip install -e ".[dev]"

Running tests
-------------

Use pytest to run all unit tests.

.. code-block:: bash

    pytest

Building docs
-------------

Generate the docs:

.. code-block::  bash

    cd docs
    make html

You can then check out the generated HTML:
    
.. code-block::  bash

    cd docs/build/html
    python3 -m http.server 


Formatting code
---------------

mia's code is formatted using `black <https://github.com/ambv/black>`_. Run the formatter as
follows:

.. code-block::  bash
    
    make format

