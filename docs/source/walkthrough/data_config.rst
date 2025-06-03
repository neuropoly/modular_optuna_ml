Data Configuration
==========================

Reviewing Your Data
-------------------

Like any good data scientist, lets start by looking at the data we will be using for our MOOP analysis:

.. csv-table:: data.csv
    :file: data.csv
    :widths: 10, 30, 30, 30, 30
    :header-rows: 1

Based on this, lets aim to run some simple pre-processing throughout the MOOPs run. This will include:

* Drop the 'meta' column, as it contains data more likely to distract the model than to inform it.
* Encoding 'bar' to be numeric with a One-Hot encoding
* Scaling the data in each column so that it abides by a Unit Norm (has a mean of 0, and a standard deviation of 1).

Lets start setting up the configuration file to do this!

Formatting the Header
---------------------

First thing's first, the config needs to specify the where the data it is configuring is, and in what format it is. If we start by copying the data config template (which can be found in the 'config_templates' directory), we can modify the header to look something like this:

.. code-block:: json

    {
        "label": "walkthrough_data",
        "data_source": "data.csv",
        "format": "tabular",
        "separator": ",",
        "index": "id",
        ...
    }

Each of these entries correspond to the following behaviour in MOOPS:

* "label": How studies involving this dataset will be identified; corresponds to the {data} part of the {data}__{model}__{study} table IDs in the final database
* "data_source": The path to the dataset you want to use for this MOOPs run, including the file name *and* its extension!
* "format": The structure of the data in the data source you specified prior. Currently only 'tabular' is supported (representing data in csv-like formats, such as csv and tsv)

Any arguments past these three will be specific to the format you specified; in our case adds two other config entries:

* "separator": The character(s) used to separate the columns in the text file.
* "index": The column to use as the index (label) for each sample. If not provided, each row's position is used instead.

.. admonition:: Why No Target?

    You may have noticed the lack of a way to designate which column in the dataset is the "target" feature; it is instead handled by the study configuration file (see :ref:`study-config-walkthrough`). Why?

    The primary reason is that subsequent MOOP runs using the same study configuration will also write their output to the same database output. If the 'target' metric were defined in the data configuration, this could result in mixing the results of analyses with different objectives (i.e. identifying an allergy to peanuts vs. post-surgical outcomes) into a single file, without it being apparent that this had happened. Therefore, while somewhat unintuitive, the 'target' of a study (and the objective function which will be used to evaluate the model's ability to predict it) was placed in the study configuration file instead.

Data Hooks and You
------------------

In MOOP, any operation which would inspect or modify any of the datasets is done via a "hook" into the analysis. Data hooks can be applied either before or after a given data split operation; the former will apply the hook to the entire dataset at that point, while the latter will attempt to use the "training" subset of the data to fit the hook before it is applied to both subsets, allowing you to mitigate potential data leakage. To demonstrate the former, lets define a data hook which will drop the "meta" column from the entire dataset. We'll apply it pre-split, to the entire dataset, by placing it in the ``pre_split_hooks`` list:

.. code-block:: json

    {
        ...
        "pre_split_hooks": [
            {
                "type": "drop_features_explicit",
                "features": ["meta"]
            }
        ],
        ...
    }

In contrast, one-hot-encoding our 'bar' column should done by fitting it to a training dataset and then applying the result to both, as to avoid information leaking between the two. Therefore it needs to be done post-split; to do so, we add it to the ``post-split-hooks`` list, like so:

.. code-block:: json

    {
        ...
        "post_split_hooks": [
            {
                "type": "one_hot_encode",
                "features": ["bar"]
            }
        ]
        ...
    }

Finally, we have the special case data scaling. This also should be fit to a training dataset, but should be run during cross-validation within each replicate as well. To let MOOP know that this is the case, we have to add the ``"run_per_cross": true`` flag to the data-hook.

.. code-block:: json

    {
        ...
        "post_split_hooks": [
            {
                "type": "one_hot_encode",
                "features": ["bar"]
            }, {
                "type": "standard_scaling",
                "run_per_cross": true
            }
        ]
    }

Well done! The configuration file for our dataset is now complete and ready to be utilized by MOOP. Assuming you followed the full tutorial, the final resulting file should look something like this:

.. code-block:: json

    {
        "label": "walkthrough_data",
        "data_source": "data.csv",
        "format": "tabular",
        "separator": ",",
        "index": "id",
        "pre_split_hooks": [
            {
                "type": "drop_features_explicit",
                "features": ["meta"]
            }
        ],
        "post_split_hooks": [
            {
                "type": "one_hot_encode",
                "features": ["bar"]
            }, {
                "type": "standard_scaling",
                "run_per_cross": true
            }
        ]
    }

.. note::

    The order you specify the data hooks within their respective lists is also the order they will be run in. As such, you should keep in mind how the data would be modified when adding new data hooks; for example, it does not make much sense to drop a column after you have modified it! The only exception to the order you specify is that pre-split hooks will always be run before post-split hooks, of course.
