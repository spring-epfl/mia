--------

**ATTENTION:** This library is not maintained at the moment due to lack of capacity. There's a plan to eventually update it, but meanwhile check out `these <https://github.com/inspire-group/membership-inference-evaluation>`_ `projects <https://github.com/inspire-group/membership-inference-evaluation>`_ for more up-to-date attacks. 

--------

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

A library for running membership inference attacks (MIA) against machine learning models. Check out
the `documentation <https://mia-lib.rtfd.io>`_.

.. description-marker-do-not-remove

These are attacks against privacy of the training data. In MIA, an attacker tries to guess whether a
given example was used during training of a target model or not, only by querying the model. See
more in the paper by `Shokri et al <https://arxiv.org/abs/1610.05820>`_. Currently, you can use the
library to evaluate the robustness of your Keras or PyTorch models to MIA.

Features:

* Implements the original shadow model `attack <https://arxiv.org/abs/1610.05820>`_
* Is customizable, can use any scikit learn's ``Estimator``-like object as a shadow or attack model
* Is tested with Keras and PyTorch

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

Shokri et al. attack
====================

See the `full runnable example
<https://github.com/spring-epfl/mia/tree/master/examples/cifar10.py>`_.  Read the details of the
attack in the `paper <https://arxiv.org/abs/1610.05820>`_.

Let ``target_model_fn()`` return the target model architecture as a scikit-like classifier. The
attack is white-box, meaning the attacker is assumed to know the architecture. Let ``NUM_CLASSES``
be the number of classes of the classification problem.

First, the attacker needs to train several *shadow models* —that mimick the target model—
on different datasets sampled from the original data distribution. The following code snippet
initializes a *shadow model bundle*, and runs the training of the shadows. For each shadow model,
``2 * SHADOW_DATASET_SIZE`` examples are sampled without replacement from the full attacker's
dataset.  Half of them will be used for control, and the other half for training of the shadow model.

.. code-block::  python

    from mia.estimators import ShadowModelBundle

    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=NUM_MODELS,
    )
    X_shadow, y_shadow = smb.fit_transform(attacker_X_train, attacker_y_train)

``fit_transform`` returns *attack data* ``X_shadow, y_shadow``. Each row in ``X_shadow`` is a
concatenated vector consisting of the prediction vector of a shadow model for an example from the
original dataset, and the example's class (one-hot encoded). Its shape is hence ``(2 *
SHADOW_DATASET_SIZE, 2 * NUM_CLASSES)``. Each label in ``y_shadow`` is zero if a corresponding
example was "out" of the training dataset of the shadow model (control), or one, if it was "in" the
training.

mia provides a class to train a bundle of attack models, one model per class. ``attack_model_fn()``
is supposed to return a scikit-like classifier that takes a vector of model predictions ``(NUM_CLASSES, )``,
and returns whether an example with these predictions was in the training, or out.

.. code-block::  python
    
    from mia.estimators import AttackModelBundle
    
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)
    amb.fit(X_shadow, y_shadow)

In place of the ``AttackModelBundle`` one can use any binary classifier that takes ``(2 *
NUM_CLASSES, )``-shape examples (as explained above, the first half of an input is the prediction
vector from a model, the second half is the true class of a corresponding example).

To evaluate the attack, one must encode the data in the above-mentioned format. Let ``target_model`` be
the target model, ``data_in`` the data (tuple ``X, y``) that was used in the training of the target model, and
``data_out`` the data that was not used in the training.
    
.. code-block::  python

    from mia.estimators import prepare_attack_data    

    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

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

