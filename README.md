# DeepTFGP:Deep Learning-Based Traffic Flow Graph Prediction

## Introduction
This is the relevant code for Paper DeepTFGP. Some code snippets come from other authors.
If you find that we are using some of your code, please write to us and we will add you to the list of acknowledgements
## Data
We provide a dataset UKNH,the data comes from publicly available data from the UK Transport Department.
Here, we provide the original csv form file, as well as our finished TFG file, available as an h5 file

train.h5 and test.h5 is the main UKNH
## Dependency

PyTorch 1.6.0

python 3.8.3

numpy

h5py

pandas

## Usage
With our dataset, you only need to run experitTFP.py 

If the run fails, it is likely a path problem, please check whether the file path in the code is consistent with the specific file path in the project

If you have any questions about how your code runs, please raise them in the issue and we'll be happy to answer them

## Hint
If you plan to collect relevant data yourself, you can put the CSV data in the data directory and use the code in Utils to generate a TFG dataset.

If you still do not clearly understand the flow of TFG production, you can raise an issue and we will try to provide more fragmentary code for reference

All in all, we will be happy to resolve your questions about this code
