###
mia
###

|pypi| |license| |build_status| |docs_status| |zenodo|

.. |pypi| image:: https://img.shields.io/pypi/v/mia.svg
   :target: https://pypi.org/project/mia/
   :alt: PyPI version

.. |build_status| image:: https://travis-ci.org/spring-epfl/mia.svg?branch=master
   :target: https://travis-ci.org/spring-epfl/mia
   :alt: Build status

.. |docs_status| image:: https://readthedocs.org/projects/mia-lib/badge/?version=latest
   :target: https://mia-lib.readthedocs.io/?badge=latest
   :alt: Documentation status

.. |license| image:: https://img.shields.io/pypi/l/mia.svg
   :target: https://pypi.org/project/mia/
   :alt: License

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1433744.svg
   :target: https://zenodo.org/record/1433744
   :alt: Citing with the Zenodo

A library for estimating vulnerability of ML models to membership inference attacks (MIA) against
machine learning models. Check out the `documentation <https://mia-lib.rtfd.io>`_.

.. description-marker-do-not-remove

=======================
What's changed in 1.0.0
=======================

The library now relies on `F-BLEAU <https://github.com/gchers/fbleau>` leakage measurement tool for
estimating the worst-case vulnerability to standard MIA attacks.

Why we removed shadow model attacks 🌱
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The `original shadow-model attack <https://arxiv.org/abs/1610.05820>` can be extremely costly as
it requires training multiple (e.g., 10—20) "shadow models" of the same architecture as the target
model. The leakage estimation approach is efficient and fast, provides more accurate worst-case
guarantees, at much lower financial and environmental costs.

.. getting-started-marker-do-not-remove

===============
Getting started
===============

You can install mia from PyPI:

.. code-block::  bash

    pip install mia

.. usage-marker-do-not-remove

=====
Usage
=====

mia takes a trained model with sklearn API, the training dataset, and a testing dataset.

.. code-block ::

>>> est = estimate_mia_vuln(clf=clf, X_train, y_train, X_test, y_test)
0.5216



.. misc-marker-do-not-remove

======
Citing
======

.. code-block::

   @misc{mia,
     author       = {Bogdan Kulynych and
                     Mohammad Yaghini},
     title        = {{mia: A library for running membership inference
                      attacks against ML models}},
     month        = sep,
     year         = 2018,
     doi          = {10.5281/zenodo.1433744},
     url          = {https://doi.org/10.5281/zenodo.1433744}
   }
