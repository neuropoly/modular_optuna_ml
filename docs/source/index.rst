.. MOOP documentation master file, created by
   sphinx-quickstart on Fri Jan 31 12:59:28 2025.

ModularOptuna: Streamlined ML Hyperparameter Tuning
===================================================

**Modular Optuna (MOOP)** is a utility package which aims to allow for streamlined and re-usable implementations of machine learning hyper-parameter tuning evaluation. As the name suggests, this package is designed as a wrapper for the `Optuna <https://optuna.readthedocs.io/en/stable/>`_ hyperparameter selection library; this tool just generalizes its use to allow for a replicable, distributable, and most importantly modular implementation.

What MOOP Can Do For You
########################

Data-Model-Study Modularity
---------------------------

In the majority of machine learning pipeline designs, the process of building, training, and evaluating the pipeline can be divide into three stages:

* **Data Preparation:** Where the data that is intended to be use is stored, and how it should be processed and provided to the pipeline.
* **Model Structure:** The type of machine learning model to use, and what hyperparameters should be used/explored.
* **Study Design:** What metrics we want measured of our model's performance, where these metrics should be recorded, and how hyperparameter tuning (if any) should be done.

MOOP stratifies each of these stages explicitly, giving them each a unique configuration file. In turn, ablation studies along these lines are made trivial; just configure the variations at each step, and iteratively test all the combinations you want to!

For further details on how this is done, and its ramifications for your analysis, please refer to **TODO**.


Robust Performance Assessment
-----------------------------

MOOP uses an extension of the classic 'train-test-validate split' approach to machine learning evaluation. Rather than relying on a single testing out-group (which can produce misleading results if it happens to not be representative of the overall sampling population), by default MOOP will generate multiple testing splits (called "replicates") and run your full analysis on each. Doing so allows for a more robust estimate of how well your model will performance when applied to data it has not seen before (often called "inference").

Note that, while these splits are effectively random, so long as the data and study config remain the same (and the latter defines a random seed), they will be consistent across re-runs, *even if the model config changes*. As a result, the performance metrics produce by each replicate can be treated as samples for the purpose of statistical analyses, and compared as such. No longer do you have to guess whether a 5% improvement is significant or not; show it with statistics!


Plug-and-Play Design
--------------------

**WORK IN PROGRESS:** This is currently not implemented in the code base, but will be implemented soon(tm), once core functionality has been implemented and tested thoroughly. To get a peak into how this will look, see the DataHook base class, alongside its associated decorator ``registered_data_hook`` for a glimpse on how this will be implemented going forward.

While MOOP strives to provide native support for most common operations (providing utilities for many SciKit-Learn processes), it is inevitable that you may need to use a custom model type, or unique method of data pre-processing, or even a custom method of evaluating a pipeline's performance. To account for this, MOOP is designed to be easily extendable through Pythonic plugins, allow for custom implementations of each of these analyses to be integrated and used on the fly!

Please see **TODO** for details on how to add this kind of custom functionality!


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction/getting_started
   introduction/faq

