#!/bin/bash

# Parse the command line arguments
INPUT_FILE=$1
OUT_FOLDER=$2
SCT_PATH=$3

# Add the SCT utilities to the PATH because it refuses to co-operate otherwise
PATH=$PATH:$SCT_PATH

# Identify attributes of the file
if [[ "$INPUT_FILE" == *"T1"* ]]; then
  CONTRAST="t1"
elif [[ "$INPUT_FILE" == *"T2"* ]]; then
  CONTRAST="t2"
else
  echo "Invalid file contrast, ending early"
  exit
fi

# Generate the output values for this segmentation
SEG_NAME="${INPUT_FILE##*/}"
SEG_NAME="${SEG_NAME%%.*}_sciseg"

SEG_FILE="$OUT_FOLDER/$SEG_NAME.nii.gz"

# Run SCI seg, a tuned version of DeepSeg, on the file
if [ ! -f "$SEG_FILE" ]; then
  sct_deepseg -i "$INPUT_FILE" -o "$OUT_FOLDER/$SEG_NAME" -task "seg_sc_lesion_t2w_sci"
else
  printf "\n"
  echo "Segmentation already exists, skipping"
fi

# Modified version of seg output to
SEG_NAME="${SEG_NAME}_sc_seg.nii.gz"

# Generate the output values for vertebral labelling
VERT_FILE="$OUT_FOLDER/${SEG_NAME%%.*}_labeled.nii.gz"

# Identify the vertebrae within the segmentation
if [ ! -f "$VERT_FILE" ]; then
  sct_label_vertebrae -i "$INPUT_FILE" -s "$SEG_FILE" -c "$CONTRAST" -ofolder "$OUT_FOLDER"
else
  printf "\n"
  echo "Vertebral labels already exist, skipping"
fi

# Generate the output values for segmentation processing
PER_VERT_OUT_NAME="sciseg_vertebrae_metrics.csv"
PER_VERT_OUT_FILE="$OUT_FOLDER/$PER_VERT_OUT_NAME"

# Use those labels alongside the segmentation to generate per-vertebrae metrics
if [ ! -f "$PER_VERT_OUT_FILE" ]; then
  echo "Beginning segmentation processing"
  OLD_DIR=$PWD
  cd "$OUT_FOLDER" || echo "Could not enter output directory for some reason; perhaps it got deleted during runtime?"
  sct_process_segmentation -i "$SEG_FILE" -vert 2:6 -vertfile "$VERT_FILE" -perlevel 1 -o "$PER_VERT_OUT_NAME"
  cd "$OLD_DIR" || echo "Could not return to original directory for some reason; no idea how you managed that!"
  echo "Finished segmentation processing"
else
  printf "\n"
  echo "Vertebral metric file already exist, skipping"
fi
