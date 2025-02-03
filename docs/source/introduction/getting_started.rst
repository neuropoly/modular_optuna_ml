Getting Started
===============

Installation
------------

1. Clone this repository to wherever you need it:
.. code-block::

    git clone https://github.com/NeuroPoly/modular_optuna_ml.git

2. Create a new Conda/Mamba environment with the dependencies needed:
.. code-block::

    conda env create -f environment.yml

    # OR

    mamba env create -f environment.yml

3. Activate said environment
.. code-block::

    conda activate modular_optuna_ml

    # OR

    mamba activate modular_optuna_ml

4. Done!

Running MOOP
------------

To run MOOP, you need 3 files:

* A data configuration file, which refers to a tabular dataset and dictates how it will be read and processed (with hyperparameter tuning, if requested)
* A model configuration file, which dictates the type of machine learning model to use, along with any hyper-parameters (to be tuned or otherwise)
* A study configuration file, which dictates where the results of the analysis should be placed and the number of replicates/crosses which should be utilized.

Once you have these, run the following to have it run in MOOP (replacing the values within the curly brackets with the path to their respective files):

.. code-block::

    python run_ml_analysis.py -d {data_config} -m {model_config} -s {study_config}

An example version of each of these configuration files is provided in the 'testing' directory of your install script. If we wanted to run a logistic regression using these configuration files, the above command would become:

.. code-block::

    python run_ml_analysis.py -d testing/iris_data/iris_config.json -m testing/model_configs/log_reg.json -s testing/testing_study_config.json


