Model Configuration
===================

Model Configuration Set-Up
----------------------------

Like before, lets first consider the model we want to tune. For the sake of simplicity, we will be tuning a classic Logistic Regression with L2 regularization. For now we'll explicitly set the L2 penalty (``l2_c``) for the sake of demonstration; skip to :ref:`model_hyper_parameter_tuning` if you want to see how to enable Optuna's hyperparameter tuning.

Let's start by first defining the header of the configuration file. This is much simpler than the data configuration model, requiring only two entries: the model type (in this case, "LogisticRegression"), and the label it should have within the database:

.. code-block:: json

    {
        "label": "tutorial_logreg",
        "model": "LogisticRegression",
        ...
    }

Now we just need to specify the parameters of the model. Unlike SciKit-Learn, MOOP requires you specify the solver explicitly, so lets do that first:

.. code-block:: json

    {
        ...
        "parameters": {
            "solver": "saga"
        }
    }

Now to specify that we want L2 regularization, and that we want the penalty to be the same as SciKit-Learn's default (1.0). This is done by simply adding the respective entries to the parameters list:

.. code-block:: json

    {
        ...
        "parameters": {
            "solver": "saga",
            "penalty": "l2",
            "l2_c": 1.0
        }
    }

Done! A very simple model config (though one without any form of automated hyperparameter tuning). If you did everything so far, your config file should look something like this:

.. code-block:: json

    {
        "label": "tutorial_logreg",
        "model": "LogisticRegression",
        "parameters": {
            "solver": "saga",
            "penalty": "l2",
            "l2_c": 1.0
        }
    }

.. _model_hyper_parameter_tuning:

Specifying Hyper-Parameter Tuning
---------------------------------

Now, while the configuration file we established prior is sufficient for the purposes of simple analyses, MOOP (through Optuna) is capable of automatically tuning these hyper-parameters for each replicate for us. Doing so is straightforward as well; just replace the constant values previously with short dictionary describing how MOOP should tune it. For example, if we want to test L1, L2, and no (null) regularization, we can update the configuration file we had prior like so:

.. code-block:: json

    {
        ...
        "parameters": {
            ...
            "penalty": {
                "label": "penalty",
                "type": "categorical",
                "choices": ["l1", "l2", null]
            },
            ...
        }
    }

And for a numeric hyper-parameter, lets specify the L1 and L2 penalties which can be tested. For now lets just have both sampled between 0.1 and 10:

.. code-block:: json

    {
        ...
        "parameters": {
            ...
            "l1_c": {
                "label": "logreg_l1",
                "type": "float",
                "low": 0.1,
                "high": 10
            },
            "l2_c": {
                "label": "logreg_l2",
                "type": "float",
                "low": 0.1,
                "high": 10
            }
        }
    }

Note that each of these "tunable" parameters requires at least two arguments:

* **label**: How MOOP should identify this hyperparameter when saving the results of an Optuna trial. Whatever you place here will be the name of the column in the database, which in turn will track how this value changes throughout the Optuna tuning process.
* **type**: The type of Optuna tunable parameter this should be. Currently MOOP supports 3 options, discussed in further detail below.

Depending on the type specified, the remaining values you need to provide will change:

* *float:* A floating point (decimal) number. Requires "low" and "high" values be specified, which determine the minimum and maximum values Optuna can select during tuning, respectively. This can be sampled on a logarithmic scale by specifying ``"log": true`` alongside these values, though this is optional.
* *int:* An integer (whole) number. Requires "low" and "high" values be specified, which determine the minimum and maximum values Optuna can select during tuning, respectively.
* *categorical:* A set of choices, specified by you within the "choices" list. The contents of this list can be anything the corresponding model accepts. At least one value must be in this list!

Now Optuna will attempt to maximize the model's performance in reach replicate by iteratively modifying any "tunable" parameters you've given it. All that's left to implement is the study parameters, which define how MOOP (and Optuna) will do so!
