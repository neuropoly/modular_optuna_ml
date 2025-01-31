What Is MOOP?
========================

*Modular Optuna Machine Learning* (*MOOP* for short) was developed to aid in the automated evaluation of machine learning pipeline performance in a robust and modular manner. This is accomplished by structuring the analysis of a machine learning pipeline to follow a few core paradigms:

Data-Model-Study Modularity
###########################

In the majority of machine learning pipelines designs, the process of building, training, and evaluating the pipeline can be divide into three stages:

* **Data Preparation:** Where the data that is intended to be use is stored, and how it should be processed and provided to the pipeline.
* **Model Structure:** The type of machine learning model to use, and what hyperparameters should be used/explored.
* **Study Design:** What metrics we want measured of our model's performance, where these metrics should be recorded, and how hyperparameter tuning (if any) should be done.

MOOP stratifies each of these stages explicitly, giving them each a unique configuration file. For further details on how this is done, and its ramifications for your analysis, please refer to **TODO**:


Repeated Train-Test-Validate Splits
#####################################

MOOP follows the standard `"train-test-validate" <https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets>`_ data splitting methodology usually employed in machine learning analyses, but extends it with repeated testing splits (referred to as 'replicates' throughout our documentation). While this can greatly extend the time required to evaluate a pipeline, doing so allows MOOPto mitigate the possibility of a single testing split being unusually difficult (or easy) for the pipeline to predict.

For more details on how MOOP handles data splitting, see **TODO**.

Highly Replicable Results
#########################

Like any good data science package, so long as you provide a random seed in the study's configuration, repeated runs of the analysis will always produce the same results. MOOPis no different here.

However, MOOP extends this further; so long as your data and study configuration is the same, the train-test-validate splits of the data are also guaranteed to be identical, regardless of the model's configuration! This allows for you to directly compare the performance metrics of multiple models directly, as they will be trained and evaluated on the *exact* same data during both cross-validation and replicates.

**PLEASE NOTE**: While running MOOP with the same packages will always be guaranteed to produce the same results, changing the version of packages that MOOP (or any of its plugins) require may still result in differing output. MOOP cannot do anything about this, as such behaviour is the result of how these packages manage their own RNG, so please ensure that your runtime environment is the same if you rely on consistent output!


Plug-and-Play Design
####################

**WORK IN PROGRESS:** This is currently not implemented in the code base, but will be implemented soon(tm), once core functionality has been implemented and tested thoroughly. However, parts of this already exist; see the DataHook base class, alongside its associated decorator ``registered_data_hook`` for a glimpse on how this will be implemented going forward.

While MOOP strives to provide native support for most common operations (providing utilities for many SciKit-Learn processes), it is inevitable that you may need to use a custom model type, or unique method of data pre-processing, or even a custom method of evaluating a pipeline's performance. To account for this, MOOP is designed to be easily extendable through Pythonic plugins, allow for custom implementations of each of these analyses to be integrated and used on the fly!

Please see **TODO** for details on how to add this kind of custom functionality!

What Can MOOP Do?
=================

Automated Train-Test-Validation Analyses
########################################

The primary function of MOOP, simply by providing a dataset and a set of configuration files, an automated analysis of the performance of your pipeline during and after hyper-parameter tuning will be run. Once its done, you can query the resulting SQLite database to review how the model's performance changed during tuning, as well as how variable its performance is across different splits of your original data.

Configurable Metric Collection
##############################

Which performance metrics you want to collect, and at what stage of pipeline tuning you want them evaluated at, is fully configurable! Simply list the metrics you want in the study configuration file, and they will be automatically calculated and tracked as the pipeline is tuned, ready for you to study and review once the analysis is complete.

Easy Study Variation
####################

As MOOP uses a modular configuration setup, modifying parts of your study without needing rewrite your entire analysis is now possible! Only modify what you need, and re-use everything else; so long as the processes remain compatible (no using a study designed for categorical analyses on a continuous task), everything will run just the same as before. This modularity makes combinatorial analyses trivial as well; just write the configuration files for each component, and run them iteratively.

What Can't MOOP Do?
===================

Data Cleaning
#############

MOOP will *not* clean your data for you. The data-cleaning hooks provided in this dataset are not meant for general use, having been modified to dynamically modify themselves on-the-fly to account for changes in data contents pre- and post-split. This makes the resulting "cleaned" datasets only valid within the context they were generated; as such, while we provide a "dump" hook for dumping these "clean" data for the purposes of debugging, you should not use it to replace your own data cleaning and pre-processing procedures!

Planned, but not Implemented
############################

Other features are planned, but not currently implemented. We hope to add them soon!

* Generate figures showing how a performance metric changes throughout the hyperparameter tuning of your pipeline (both overall and on a per-replicate).
* Compared the results of multiple MOOP runs, though both statistical analyses (i.e. Kruskal-Wallace) and comparison figures.
* Utilize the results of a MOOP run to generate standalone machine learning pipelines, which can be applied and deployed without requiring MOOP or Optuna.
