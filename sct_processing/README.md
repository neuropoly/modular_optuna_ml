# Predicting Post-Surgical Outcomes for Patients with DCM

The scripts, data, and analyses presented in this folder were used to generate and test the performance of various classical ML models tasked with predicting the post-surgical outcomes of patients with DCM 1 year after surgery.

## Method Overview

1. Clinical diagnostic and MRI sequences of various sequencing modalities (Sagittal T1, Sagittal T2, Axial T2) were obtained from patients across Alberta.
2. MRI sequences were automatically segmented using the Contrast Agnostic DeepSeg algorithm provided by the Spinal Cord Toolbox (SCT) release 6.5.
3. The resulting segmentations were then used to estimate the spinal cord morphological metrics of patients in the dataset, using the `sct_process_segmentation` script provided by SCT.
    * Steps 2 and 3 were completed using `iterative_sct.py` and `contrast_agnostic_process.sh`
      * `iterative_sct.py` is a replacement for SCT's `sct_run_batch` command, set to run for every _MRI_ sequence found within a BIDS-like subject directory (rather than for each subject directory). 
        * This is an optimization for running on highly-parallel systems, as iterating by directory requires more boolean logic and error checking than doing so by file. If you have a BIDS-like dataset + a `sct_run_batch` script already, its output will work just as well.
      * `contrast_agnostic_process.sh` runs SCT's `sct_deepseg` command with the `seg_sc_contrast_agnostic` task, processing the resulting segmentation to produce estimates spinal-cord morphometrics from the `C2` to `C6` level of the spine.
4. The resulting metrics are collected and cleaned using the code provided in `softseg_data/data_cleaning.ipynb`. This process is summarized below:
    * Data in all resulting files are concatenated together into a single csv file, with the resulting duplicate headers stripped
    * Each metric file's name is parsed to identify which patient was associated with each MRI sequence, as well as the MRI sequences orientation (Axial or Sagittal), contrast ('weight', T1 or T2), and run (if the patient had multiple runs of the same MRI orientation/contrast modality)
      * If a patient has multiple runs for a given MRI modality, only the last run is used.
    * Remove descriptive-only features from the dataset which are not relevant for ML analysis
    * Pivot the vertebral levels to be treated as distinct features, rather than distinct samples
    * Calculate the Hirabayashi Recovery Ratio (HRR) for each patient, using the data contained within the BIDS dataset's `participants` file.
      * HRR is the ratio of patient improvement ('post-operative mJOA' - 'initial mJOA') over the potential improvement that patient could have seen ('Max mJOA Score (18)' - 'initial mJOA')
      * Any patients which lack either an initial or post-surgical mJOA are dropped at this point as well, as both are needed for calculating the HRR
    * Whether a patient's HRR is greater than or equal to 0.5 is calculated; this threshold has been found to indicate a clinically significant improvement [(ref)](https://pubmed.ncbi.nlm.nih.gov/23942607/), and will be the prediction target for our ML analyses.
    * Any patients which did not undergo surgical treatment are dropped from the dataset
    * Any features collected after surgery (aside from the aforementioned patient HRR) are dropped, as it would not be available to the ML model when it makes its prediction.
    * Three kinds of datasets are generated from combining these datasets together
      * Imaging only: contains only morphometrics derived from a single MRI modality (MRI orientation + contrast ) for each patient which had an MRI sequence for that modality (i.e. spincal cord diameter, angle, etc.). One sub-set is generated per modality (3 total)
      * Clinical only: contains only non-imaging derived metrics (i.e. sex, age, symptom severity etc.) for each patient.
      * Full: Contains both of the prior for each patient, for each MRI modality the patient had. One subset is generated per modality (3 total) 
5. Dataset configurations files are generated for each dataset via `softseg_data/auto_configs.ipynb`:
   * All datasets have the following pre-processing methods included:
     * Explicit dropping of redundant meta-features (such as where the data was acquired) and features which would not be available pre-surgery (such as whether the patient follow-up up 1 month later)
     * Any features which, post in-out split, are missing more than 50% of their sample values are dropped
     * Any samples which, post in-out split, are missing more than 50% of their features are dropped
     * All features undergo standardization to ensure a similar range and distribution
   * For all datasets with clinical metrics:
     * Categorical columns are imputed by filling nulls w/ the mode value, then one-hot encoded, prior to standardization
   * 5 data configuration permutations were generated for each dataset:
     * No further pre-processing
     * Recursive Feature Elimination (RFE) to select a subset of the available features
     * Principal Component Analysis (PCA) to transform the available features into a variance-conserving space
     * RFE, followed by PCA, applied to the available features
     * PCA, followed by RFE, applied to the available features
       * **NOTE:** There is currently a bug in this specific combination, as RFE throws an error if PCA produces only one component (See [Issue #10](https://github.com/SomeoneInParticular/dcm-classic-ml/issues/10)).
6. Using the model configurations already tested in the main library, a combinatorial analysis of all combinations of dataset and ML model architecture is run:
   * Each study is run with 10 replicates, 100 trials per replicate, and 10-fold cross-validation during trial assessment.
   * Possible data permutation include:
     * Data subset
       * Imaging-only
       * Clinical-only (in this case all imaging-related "permutation" labels are filled with "none")
       * Full 
     * MRI modality:
       * T1 Sagittal
       * T2 Sagittal
       * T2 Axial
     * Data Pre-Processing
       * None
       * RFE only
       * PCA only
       * RFE into PCA
       * PCA into RFE
     * ML Model Type
       * Logistic Regression
       * Support Vector Machine
       * K-Nearest Neighbor
       * Random Forest
       * AdaBoost
   * 175 total permutations
     * While Issue #10 remains, this will be closer to 140 permutations, as 35 of them rely on said pre-processing method. With the random seed we chose, 155 permutations ran to completion.

## Results Interpretation

See `result/README.md` for further details.

