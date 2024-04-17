# ICCS2024
The code for "A Multi-Domain Multi-Task Approach for Feature Selection from Bulk RNA Datasets" paper accepted to  ICCS 2024 (International Conference on Computational Science)

## Requirements:
Python 3.8.13

CUDA Version: 12.2

## Installation

Create a new directory for your project, e.g 'ICCS2024_Karim_Salta', and navigate into it:
```bash
mkdir ICCS2024_Karim_Salta
cd ICCS2024_Karim_Salta
```

Create a virtual environment to isolate your project dependencies, e.g. named 'iccs2024env', and activate it:

python3 -m venv iccs2024KSenv

source iccs2024KSenv/bin/activate

Clone this repositary and navigate into it:

git clone https://github.com/kkarimov/ICCS2024.git

cd ICCS2024

Populate the environment:

pip install -r requirements.txt

## If you run into this issue:
ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: https://github.com/urllib3/urllib3/issues/2168' downgrading urllib package should be a good fix:

pip install urllib3==1.26.15

## Data used in this paper is proprietory, but you can run traing experiment with your own data and custom dataloader


## Data includes only two label groups. If ou have more labels you might want to update:

 - 'Network.criterionCls' from nn.BCELoss() to nn.CrossEntropyLoss()

 - 'model.FC_Classifier.reduction' layer to the one compatible with nn.CrossEntropyLoss()


## Examples

One-domain experiments, skip if no data avaialble:

trainOne.ipynb

Plot losses for one run, runnable without data:

plotLosses.ipynb

Multi-domain training, skip if no data avaialble:

trainAll.ipynb

Plot distributions, runnable without data:

processWeightsExample.ipynb

Plot overlapped features, runnable without data:

overlapAnalysis.ipynb
