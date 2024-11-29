#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --time 12:00:00
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kalum.ost

# Reset to the base environment; otherwise stupidity ensues
source activate base

# Then enter the environment we actually want
conda activate classic_ml_reloaded

# Run the script
python ~/classic_ml_reloaded/run_ml_analysis.py -d ~/classic_ml_reloaded/sct_processing/softseg_data/sag_T2w_rfe.json -m ~/classic_ml_reloaded/sct_processing/models/svc.json -s ~/classic_ml_reloaded/sct_processing/study/study_config.json --overwrite