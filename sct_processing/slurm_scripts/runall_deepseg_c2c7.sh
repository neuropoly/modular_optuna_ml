#!/bin/bash
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --time 12:00:00
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --array=0-149 # <-- 5 model permutations * 30 dataset permutations, 0 indexed, inclusive
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kalum.ost

# Purge any loaded modules
module purge

# Reset to the base environment; otherwise stupidity ensues
source activate base
# Then enter the environment we actually want
conda activate classic_ml_reloaded

# Enter the SCT Processing directory:
OLD_DIR=$PWD
# CHANGE THIS TO MATCH YOUR INSTALL PATH
cd ~/classic_ml_reloaded/sct_processing

# Get the number of data files, as its the basis of our modulo
N_DATA_CONFIGS=`ls deepseg_data/c2c7/*.json | wc -l`

# Get the data file to use via modulo
DATA_OFFSET=$(($SLURM_ARRAY_TASK_ID % $N_DATA_CONFIGS + 1))
DATA_FILE=`ls deepseg_data/c2c7/*.json | head -n $DATA_OFFSET | tail -n 1`

# Get the model file to use via divide
MODEL_IDX=$(($SLURM_ARRAY_TASK_ID / $N_DATA_CONFIGS + 1))
MODEL_FILE=`ls models/*.json | head -n $MODEL_IDX | tail -n 1`

# All jobs in this script use the same study configuration
STUDY_FILE="study/study_config.json"

echo "../run_ml_analysis.py -d $DATA_FILE -m $MODEL_FILE -s $STUDY_FILE --overwrite"

python ../run_ml_analysis.py -d "$DATA_FILE" -m "$MODEL_FILE" -s "$STUDY_FILE" --overwrite
