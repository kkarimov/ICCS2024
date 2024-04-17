# Author: Karim Salta (Karimov)
# contacts: kkarimov@gmail.com, https://github.com/kkarimov

# This loader is data-specific, needs to be updated for you data
# THe general structure of the custom loader should be the following
# class RNA_Dataset(Dataset):
#     def __init__(input1, input2, ...):
#         # your code here        
#     def _download(self, domain, use_cuda, shuffleTrain, shuffleValid):
#         # your code here
#         dataTrain = DataLoader(TensorDataset(xTrain, yTrain), batch_size=xTrain.shape[0], drop_last=True, shuffle=shuffleTrain)
#         dataValid = DataLoader(TensorDataset(xValid, yValid), batch_size=xValid.shape[0], drop_last=True, shuffle=shuffleValid)
#         return dataTrain, dataValid


from torch import from_numpy
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from filelock import FileLock

from orthrus.preprocessing.imputation import HalfMinimum
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict

class RNA_Dataset(Dataset):

    def __init__(self, PATH, DATA_FILES, DOMAINS, zscore, trainPerc, seed):

        self.PATH = PATH
        self.DOMAINS = DOMAINS
        self.DATA_FILES = DATA_FILES
        self.features = pd.read_csv(PATH + "genesCommon.txt", sep='\t', header=(0))['genes'].tolist()
        self.zscore = zscore
        self.trainPerc = trainPerc
        self.seed = seed

    def _load_data(self, domain):

        with FileLock(self.PATH+self.DATA_FILES[domain] + '.lock'):
            data = pd.read_csv(self.PATH+self.DATA_FILES[domain])
        self.samples = data['Unnamed: 0']

        return data
    
    def _normalize(self, data):

        if self.zscore:
            training_transform = make_pipeline(HalfMinimum(missing_values=0), StandardScaler())
            data[data.columns[1:-2]] = training_transform.fit_transform(data[data.columns[1:-2]].values)

        return data
    
    def _train_test_split(self, data):

        self.labels = data['label'].values
        self.labelsSet = np.unique(self.labels)
        xTrain, xValid, yTrain, yValid = train_test_split(data, self.labels, train_size=self.trainPerc, stratify=self.labels, random_state=self.seed)
        self.trainSamples, self.validSamples = xTrain['Unnamed: 0'], xValid['Unnamed: 0']

        return xTrain, xValid, yTrain, yValid

    def _download(self, domain, shuffleTrain, shuffleValid):

        data = self._load_data(domain)
        data = self._normalize(data)
        xTrain, xValid, yTrain, yValid = self._train_test_split(data)

        xTrain = from_numpy(xTrain[data.columns[1:-2]].values).float()
        xValid = from_numpy(xValid[data.columns[1:-2]].values).float()
        yTrain = from_numpy(yTrain).float()
        yValid = from_numpy(yValid).float()

        dataTrain = DataLoader(TensorDataset(xTrain, yTrain), batch_size=xTrain.shape[0], drop_last=True, shuffle=shuffleTrain)
        dataValid = DataLoader(TensorDataset(xValid, yValid), batch_size=xValid.shape[0], drop_last=True, shuffle=shuffleValid)

        return dataTrain, dataValid