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
        "index": "id"
        ...
    }

Each of these entries correspond to the following behaviour in MOOPS:

* "label": How studies involving this dataset will be identified; corresponds to the {data} part of the {data}__{model}__{study} table IDs in the final database
* "data_source": The path to the dataset you want to use for this MOOPs run, including the file name *and* its extension!
* "format": The structure of the data in the data source you specified prior. Currently only 'tabular' is supported (representing data in csv-like formats, such as csv and tsv)

Any arguments past these three will be specific to the format you specified; this is specific to each format type, and in our case adds two other config entries:

* "separator": The character(s) used to separate the columns in the text file.
* "index": The column to use as the index (label) for each sample. If not provided, each row's position is used instead.

.. admonition:: Why No Target?

    You may have noticed the lack of a way to designate which column in the dataset is the "target" feature; it is instead handled by the study configuration file (see **TODO**). Why?

    The primary reason is that subsequent MOOP runs using the same study configuration will also write their output to the same database output. If the 'target' metric were defined in the data configuration, this could result in mixing the results of analyses with different objectives (i.e. identifying an allergy to peanuts vs. post-surgical outcomes) into a single file, without it being apparent that this had happened. Therefore, while somewhat unintuitive, the 'target' of a study (and the objective function which will be used to evaluate the model's ability to predict it) was placed in the study configuration file instead.

Data Hooks and You
------------------