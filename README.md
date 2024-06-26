<table align="center">
  <tr>
    <td align="center">
      <a href="https://www.iccs-meeting.org/iccs2024/" style="font-size: 65px; text-decoration: none; color: black; style="vertical-align: middle;"">ICCS 2024:</a><br>
      International Conference on Computational Science,<br>
      Malaga, Spain, 2-4 July 2024.
    </td>
    <td><img src="https://raw.githubusercontent.com/kkarimov/ICCS2024/main/images/conference.jpeg" width="450" alt="ICCS 2024"/></td>
  </tr>
</table>

<p>&nbsp;</p>

The code for [**_"A Multi-Domain Multi-Task Approach for Feature Selection from Bulk RNA Datasets"_**](https://doi.org/10.1007/978-3-031-63772-8_3) paper accepted to [ICCS 2024](https://easychair.org/smart-program/ICCS2024/2024-07-04.html#talk:256143)
=======

<p align="center">
  <img src="https://raw.githubusercontent.com/kkarimov/ICCS2024/main/images/diagram.png" alt="Logo">
</p>

<p>&nbsp;</p>

## License
This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/) or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Please cite as (bibtex when published):

```bibtex
@InProceedings{10.1007/978-3-031-63772-8_3,
author="Salta, Karim
and Ghosh, Tomojit
and Kirby, Michael",
editor="Franco, Leonardo
and de Mulatier, Cl{\'e}lia
and Paszynski, Maciej
and Krzhizhanovskaya, Valeria V.
and Dongarra, Jack J.
and Sloot, Peter M. A.",
title="A Multi-domain Multi-task Approach for Feature Selection from Bulk RNA Datasets",
booktitle="Computational Science -- ICCS 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="31--45",
abstract="In this paper a multi-domain multi-task algorithm for feature selection in bulk RNAseq data is proposed. Two datasets are investigated arising from mouse host immune response to Salmonella infection. Data is collected from several strains of collaborative cross mice. Samples from the spleen and liver serve as the two domains. Several machine learning experiments are conducted and the small subset of discriminative across domains features have been extracted in each case. The algorithm proves viable and underlines the benefits of across domain feature selection by extracting new subset of discriminative features which couldn't be extracted only by one-domain approach.",
isbn="978-3-031-63772-8"
}
```

<p>&nbsp;</p>

## **IMPORTANT**: 
- Data used in this paper is proprietary, but you can always run the training with your own data and the custom dataloader!
- Data includes only two label groups. If you have more labels you might want to update:
  - 'src.Network.criterionCls' from nn.BCELoss() to nn.CrossEntropyLoss()
  - 'src.model.FC_Classifier.reduction' layer to the one compatible with nn.CrossEntropyLoss()

<p>&nbsp;</p>

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
#### If you run into this issue:
```error
ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: https://github.com/urllib3/urllib3/issues/2168'
```
#### downgrading urllib package should be a good fix:
```bash
pip install urllib3==1.26.15
```

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
