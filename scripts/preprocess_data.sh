#!/usr/bin/env bash

# Can execute script from anywhere
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../..

PYTHONPATH=./src/ python src/data_loaders/yelp_dataset.py

