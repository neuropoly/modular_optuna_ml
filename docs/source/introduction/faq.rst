F.A.Q.
======

What Won't MOOP Do?
###################

Data Prep/Cleaning
------------------

MOOP will *not* clean your data for you. The data-cleaning hooks provided in this dataset are not meant for general use, having been modified to dynamically modify themselves on-the-fly to account for changes in data contents pre- and post-split. This makes the resulting "cleaned" datasets only valid within the context they were generated; as such, while we provide a "dump" hook for dumping these "clean" data for the purposes of debugging, you should not use it to replace your own data cleaning and pre-processing procedures!


Permutation/Ablation Studies
----------------------------

MOOP will not run these for you itself. However, by creating sets of configuration files, or making slight variations on existing ones, you can effectively run both quickly through iterative combinatorial loops. Just make sure each config has a unique label, and MOOP will save each variation to its own result table in the output database.


What's Planned?
###############

Other features are planned, but not currently implemented. We hope to add them soon!

* Generate figures showing how a performance metric changes throughout the hyperparameter tuning of your pipeline (both overall and on a per-replicate).
* Compared the results of multiple MOOP runs, though both statistical analyses (i.e. Kruskal-Wallace) and comparison figures.
* Utilize the results of a MOOP run to generate standalone machine learning pipelines, which can be applied and deployed without requiring MOOP or Optuna.
