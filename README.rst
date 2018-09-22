###
mia
###

|build_status| |docs_status|

.. |build_status| image:: https://api.travis-ci.org/spring-epfl/hippiepug.svg?branch=master
   :target: https://travis-ci.org/spring-epfl/hippiepug
   :alt: Build status

.. |docs_status| image:: https://readthedocs.org/projects/hippiepug/badge/?version=latest
   :target: https://hippiepug.readthedocs.io/?badge=latest
   :alt: Documentation status

.. description-marker-do-not-remove

A library for running membership inference attacks (MIA) against machine learning models. Implements
the original shadow model attack by `Shokri et al. <https://arxiv.org/abs/1610.05820>`_

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

