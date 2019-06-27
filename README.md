# MeanSum: A Model for Unsupervised Neural Multi-Document Abstractive Summarization

Corresponding paper, accepted to ICML 2019: [https://arxiv.org/abs/1810.05739](https://arxiv.org/abs/1810.05739).


## Requirements

Main requirements:
- python 3.6
- torch 0.4.0

Rest of python packages in ```scripts/requirements.txt```.
Tested in Docker, image = ```pytorch/pytorch:0.4_cuda9_cudnn7```.

## Setup
1. All scripts must be executed from the scripts directory:
```
cd scripts/
```
2. Create directories that aren't part of the Git repo (checkpoints/, outputs/):
```
bash setup_dirs.sh
```
3. Download subword tokenizer and pretrained models:
```
bash download_pretrained_models.sh
```
4. Install dependencies:
```
# Build Docker image:
docker build -t stepgazaille/meansum .

# Or, execute the following scripts:
bash install_python_pkgs.sh
python update_tensorboard.py
```
5. Download the [Yelp data set](https://www.yelp.com/dataset) and place files in ```datasets/yelp_dataset/```. For example:
```
mkdir ~/downloads/yelp_dataset/
tar -C ~/downloads/yelp_dataset/ -xvf ~/downloads/yelp_dataset.tar
mv -v  ~/downloads/yelp_dataset/* ~/meansum/datasets/yelp_dataset/
```
6. Run script to preprocess the Yelp data set and create train, val and test splits:
```
# Using docker container, for example:
docker run --runtime=nvidia -it --rm \
    -v $(realpath ~/meansum):/home/meansum \
    stepgazaille/meansum \
    /bin/bash -c "cd meansum/scripts/ && bash preprocess_data.sh"

# Or locally:
bash preprocess_data.sh
```
7. Download reference summaries from [here](https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv).
Each row contains "Input.business_id", "Input.original_review_\<num\>\_id", 
"Input.original_review__\<num\>\_", "Answer.summary", etc. The "Answer.summary" is the
reference summary written by the Mechanical Turk worker.

## Usage
### Run docker container
Execute the following command to run the MeanSum docker container and access its terminal:
```
docker run --runtime=nvidia -it --rm \
    -v $(realpath ~/meansum):/home/meansum \
    stepgazaille/meansum \
    /bin/bash
```

### Evaluate pretrained model
Testing with pretrained mode. This will output and save the automated metrics. 
Results will be in ```outputs/eval/yelp/n_docs_8/unsup_<run_name>```

NOTE: Unlike some conventions, 'gpus' option here represents the GPU ID (the one which is visible) and NOT the number of GPUs. Hence, for a machine with a single GPU, you will give gpus=0
```
python train_sum.py --mode=test --gpus=0 --batch_size=16 --notes=<run_name>
```

Training summarization model (using pre-trained language model and default hyperparams).
The automated metrics results will be in ```checkpoints/sum/mlstm/yelp/<hparams>_<additional_notes>```:
```
python train_sum.py --batch_size=16 --gpus=0,1,2,3 --notes=<additional_notes> 
```
### Build subword encoder
Create a vocabulary of size 32000 from Yelp corpus:
```
PYTHONPATH=. python data_loaders/build_subword_encoder.py \
    --dataset=yelp \
    --target_size=32000 \
    --output_dir=datasets/yelp_dataset/processed/ \
    --output_fn=subwordenc
```

### Pretrain a Language Model
Pretrain a language model on Yelp corpus:
```
python pretrain_lm.py \
    --dataset=yelp \
    --save_model_fn=lm
```


### Train a Summarization Model
Pretrain a summarization model on Yelp corpus:
```
python train_sum.py \
    --load_lm=../stable_checkpoints/lm/mlstm/yelp/batch_size_512-lm_lr_0.001-notes_data260_fixed/lm_e24_2.88.pt \
    --load_clf=../stable_checkpoints/clf/cnn/yelp/batch_size_256-notes_data260_fixed/clf_e10_l0.6760_a0.7092.pt \
    --batch_size=8
```
