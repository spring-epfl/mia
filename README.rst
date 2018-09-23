###
mia
###

|build_status| |docs_status|

.. |build_status| image:: https://api.travis-ci.org/bogdan-kulynych/mia.svg?branch=master
   :target: https://travis-ci.org/bogdan-kulynych/mia
   :alt: Build status

.. |docs_status| image:: https://readthedocs.org/projects/mia-lib/badge/?version=latest
   :target: https://mia-lib.readthedocs.io/?badge=latest
   :alt: Documentation status

.. description-marker-do-not-remove

A library for running membership inference attacks (MIA) against machine learning models.

 * Implements the original shadow model attack by `Shokri et al. <https://arxiv.org/abs/1610.05820>`_
 * Customizable, can use any sklearn-like object as a shadow or attack model
 * Tested with Keras and PyTorch

.. getting-started-marker-do-not-remove

===============
Getting started
===============

Clone the repo and install using pip:

.. code-block::  bash

    pip install -e ".[dev]"

Then, you can run tests:

.. code-block::  bash

    pytest

