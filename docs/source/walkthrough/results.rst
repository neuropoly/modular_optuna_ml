Getting and Interpreting Results
================================

Assuming you now have a data, model, and study configuration file, running MOOP is as simple as running the following command:

.. code-block:: bash

    python run_ml_analysis.py -d data_config.json -m model_config.json -s study_config.json

MOOP will handle the rest, outputting and saving the results to the SQLite database you specified in the study config.

.. note::

    By default MOOP will not overwrite existing results placed in the same output file; this is to prevent accidentally deleting the results of any analyses you ran prior which you might have want to keep. To get around this, you can either move the results file to a new location (allowing MOOP to generate a new results file with the newer analyses results in its place), or add the `--overwrite` flag to prior command. The latter will delete the prior results as MOOP initiates, however, so be careful!


Interpreting the Results
------------------------

.. attention::

    Currently, all result interpretation must be done manually by you! This tutorial will document common analyses in that vein, but you should consider alternative solutions depending on your use case.

    MOOP is also still in development, with "automated" methods of result interpretation and visualization planned. As such, like the code itself, these tutorials are subject to change; apologies in advance!

Pulling Results from the DataBase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once MOOP has completed its study, you should now have a SQLite database file placed in the location you requested in the study configuration file. Naturally, the first step to interpreting the results is to load them; we'll use Pandas for this. Assuming you have not changed your directory after you finished the MOOP run, the following should load the results of each MOOP run into a Pandas DataFrame, with one row per study:

.. code-block:: python

    from sqlite3 import connect
    import pandas as pd

    db_con = connect("output/moop_results.db")
    moop_runs = pd.read_sql(
        "SELECT * FROM sqlite_master",
        con=con
    ).loc[:, 'name']

We now have a list of all MOOP analyses contained within the database. If you ran MOOP using the configuration files we built prior, one of the entries should be
``tutorial_study__tutorial_logreg__walkthrough_data``; the three labels we defined joined with '``__``'. To load the results of that specific run, another SQLite call is needed:

.. code-block:: python

    tutorial_df = pd.read_sql(
        f"SELECT * FROM 'tutorial_study__tutorial_logreg__walkthrough_data'",
        con=con
    )

``tutorial_df`` will now contain measurements MOOP took through its analysis. It will look something like this:

.. csv-table:: Results
    :file: results.csv
    :widths: 30, 30, 30, 30, 30, 30
    :header-rows: 1



Single-Run Analysis
^^^^^^^^^^^^^^^^^^^

Assuming you use configuration files with the labels we defined previously, one of the entries in "moop_runs" Series should be "walkthrough_data__tutorial_logreg__tutorial_study". To load the results of that specific MOOPs run, we can run the following:

adfasdfadsf