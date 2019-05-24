#!/usr/bin/env bash

# Can execute script from anywhere
PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd $PARENT_PATH
cd ../..


#Download subword tokenizer built on Yelp:
wget -P datasets/yelp_dataset/processed/ https://s3.us-east-2.amazonaws.com/unsup-sum/subwordenc_32000_maxrevs260_fixed.pkl

# Download summarization model 
SUM_MODEL_PATH="stable_checkpoints/sum/mlstm/yelp/batch_size_16-notes_cycloss_honly-sum_lr_0.0005-tau_2.0/"
mkdir -p $SUM_MODEL_PATH
wget -P $SUM_MODEL_PATH https://s3.us-east-2.amazonaws.com/unsup-sum/sum_e0_tot3.32_r1f0.27.pt

# Download language model
LANG_MODEL_PATH="stable_checkpoints/lm/mlstm/yelp/batch_size_512-lm_lr_0.001-notes_data260_fixed/"
mkdir -p $LANG_MODEL_PATH
wget -P $LANG_MODEL_PATH https://s3.us-east-2.amazonaws.com/unsup-sum/lm_e24_2.88.pt

# Download classification model
CLASS_MODEL_PATH="stable_checkpoints/clf/cnn/yelp/batch_size_256-notes_data260_fixed/"
mkdir -p $CLASS_MODEL_PATH
wget -P $CLASS_MODEL_PATH https://s3.us-east-2.amazonaws.com/unsup-sum/clf_e10_l0.6760_a0.7092.pt

