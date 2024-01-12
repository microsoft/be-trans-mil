#!/bin/bash

# This script creates a folder of symbolic links to the slides for a given dataset.

# This script assumes that the dataset is in the `cyted-raw-20221102` folder of the `datasets` container.
# The dataset is expected to be a TSV file with a column for the H&E slide (column 13),
# TFF3 slide (column 14), and P53 slide (solumn 15).
# The script will create a folder for each of the three slide types (one folder for H&E slides, one for TFF3, etc),
# and create a symbolic link to the slide in its original location in the `cyted-raw-20221102` folder.

# Setup:
# Please follow the instructions in docs/workflow.md to mount the `datasets` container of your Azure storage account.
# After those setups, the `datasets` container of the storage account will be mounted at /cyted


TEMP=~/cyted-temp
mkdir $TEMP
# Modify this to point to the dataset you want to process
DATA=/cyted/cyted-raw-20221102
# Modify this to point to the TSV file you want to process
INPUT_FILE=$DATA/exclude_wrong_stains_2022-11-22.tsv

# Exclude all slides with QC Report: "fail". "Fail" is a string that only occurs in the QC Report column,
# so we can use grep directly
NO_FAIL_SUBSET=$TEMP/exclude_wrong_stains_2022-11-22_no_fail.tsv
echo "Excluding slides with 'QC Report: fail'"
grep -v "fail" $INPUT_FILE > $NO_FAIL_SUBSET

# Cut out the columns for H&E, TFF3, P53, and discard the column header. Remove all rows that contain "NA"
echo "Reading H&E / TFF3 / P53 columns from data file"
SUFFIX=_slides_only
HE_FILE=$TEMP/he$SUFFIX.txt
TFF3_FILE=$TEMP/tff3$SUFFIX.txt
P53_FILE=$TEMP/p53$SUFFIX.txt
cut -f 13 $NO_FAIL_SUBSET | tail -n +2 | grep -v NA > $HE_FILE
cut -f 14 $NO_FAIL_SUBSET | tail -n +2 | grep -v NA > $TFF3_FILE
cut -f 15 $NO_FAIL_SUBSET | tail -n +2 | grep -v NA > $P53_FILE

HE_FOLDER=$TEMP/he$SUFFIX
TFF3_FOLDER=$TEMP/tff3$SUFFIX
P53_FOLDER=$TEMP/p53$SUFFIX
echo "Creating folders for results"
mkdir $HE_FOLDER
mkdir $TFF3_FOLDER
mkdir $P53_FOLDER
echo "Creating links for H&E slides in $GROUPED/he"
cat $HE_FILE | xargs -I {} ln -s $DATA/{} $HE_FOLDER/{}
echo "Creating links for TFF3 slides in $GROUPED/tff3"
cat $TFF3_FILE | xargs -I {} ln -s $DATA/{} $TFF3_FOLDER/{}
echo "Creating links for P53 slides in $GROUPED/p53"
cat $P53_FILE | xargs -I {} ln -s $DATA/{} $P53_FOLDER/{}
