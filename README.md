# International Conference on Computational Science, Malaga, Spain, 2-4 July 2024 ([ICCS 2024](https://www.iccs-meeting.org/iccs2024/)).
The code for "A Multi-Domain Multi-Task Approach for Feature Selection from Bulk RNA Datasets" paper accepted to  ICCS 2024 
=======
<p align="center">
  <img src="https://raw.githubusercontent.com/kkarimov/ICCS2024/main/images/diagram.png" alt="Logo">
</p>
<p align="center">
  <img src="https://github.com/kkarimov/ICCS2024/blob/main/images/groupped1.png" alt="" style="width: 31.2%; margin: 0px; display: inline;">
  <img src="https://github.com/kkarimov/ICCS2024/blob/main/images/groupped2.png" alt="" style="width: 31.2%; margin: 0px; display: inline;">
  <img src="https://github.com/kkarimov/ICCS2024/blob/main/images/groupped3.png" alt="" style="width: 31.2%; margin: 0px; display: inline;">
</p>


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
```bash
python3 -m venv iccs2024KSenv
source iccs2024KSenv/bin/activate
```
Clone this repositary and navigate into it:
```bash
git clone https://github.com/kkarimov/ICCS2024.git
cd ICCS2024
```
Populate the environment:
```bash
pip install -r requirements.txt
```
## If you run into this issue:
```error
ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: https://github.com/urllib3/urllib3/issues/2168'
```
## downgrading urllib package should be a good fix:
```bash
pip install urllib3==1.26.15
```

<p>&nbsp;</p>

## **IMPORTANT**: Data used in this paper is proprietory, but you can always run the training with your own data and the custom dataloader!

<p>&nbsp;</p>

## Data includes only two label groups. If you have more labels you might want to update:

 - 'src.Network.criterionCls' from nn.BCELoss() to nn.CrossEntropyLoss()

 - 'src.model.FC_Classifier.reduction' layer to the one compatible with nn.CrossEntropyLoss()

<p>&nbsp;</p>

## Examples:

One-domain experiments, __skip__ if __no data avaialble__

[trainOne.ipynb](https://github.com/kkarimov/ICCS2024/blob/main/trainOne.ipynb)

Plot losses for one run, __runnable without data__

[plotLosses.ipynb](https://github.com/kkarimov/ICCS2024/blob/main/plotLosses.ipynb)

Multi-domain training, __skip__ if __no data avaialble__

[trainAll.ipynb](https://github.com/kkarimov/ICCS2024/blob/main/trainAll.ipynb)

Plot distributions, __runnable without data__

[processWeightsExample.ipynb](https://github.com/kkarimov/ICCS2024/blob/main/processWeightsExample.ipynb)

Plot overlapped features, __runnable without data__

[overlapAnalysis.ipynb](https://github.com/kkarimov/ICCS2024/blob/main/overlapAnalysis.ipynb)
