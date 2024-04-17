# Author: Karim Salta (Karimov)
# contacts: kkarimov@gmail.com, https://github.com/kkarimov

from src.model import *
from torch import optim

class Network(object):
    def __init__(self, DOMAINS, use_cuda,
            inputDim, latentDim, n_hidden, learningRateAE, learningRateD, weightDecay, alpha, beta, lambd, gamma, 
            classTrain, AETrain, sparse):
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.gamma = gamma
        self.classTrain = classTrain
        self.AETrain = AETrain
        self.sparse = sparse
        self.domains = DOMAINS
        self.inputDim = inputDim


        self.OTO = OTOLayer(self.inputDim).to(self.device)
        self.net, self.optNet = dict(), dict()
        for _ in DOMAINS:
            self.net[_] = FC_VAE(self.inputDim, latentDim, n_hidden, self.sparse).to(self.device)
            self.optNet[_] = optim.Adam(list(self.net[_].parameters()), lr=learningRateAE)
        self.netCondClf = FC_Classifier(latentDim).to(self.device)
        self.optOTO = optim.Adam(list(self.OTO.parameters()), lr=learningRateAE)
        self.optNetCondClf = optim.Adam(list(self.netCondClf.parameters()), lr=learningRateD, weight_decay=weightDecay)
        self.criterionRec = nn.MSELoss()
        
        # We use binary crossentropy
        self.criterionCls = nn.BCELoss()
        #  If you have more than two label groups you might want to edit forward and use nn.CrossEntropyLoss()

    def criterionKL(self, mu, logvar, epoch, lambd):
        if lambd>0:
            KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return lambd * KLloss * torch.exp(torch.FloatTensor([-epoch/10000]).to(self.device))
        return 0
    
    def getLatents(self, domainInputs, labelInputs, domain):
        self.OTO.eval()
        domainInputs, labelInputs = Variable(domainInputs).to(self.device), Variable(labelInputs).to(self.device)
        x = self.OTO(domainInputs)
        self.net[domain].eval()
        domainRecon, domainLatents, domainMu, domainLogvar = self.net[domain](x)
        return domainMu # actually it might be domainLatents

    def trainIter(self, inputs, epoch): 
        self.OTO.train()
        self.OTO.zero_grad()
        for _ in self.domains:
            self.net[_].train()
            self.net[_].zero_grad()
        self.netCondClf.train()
        self.netCondClf.zero_grad()
        inputsNew, temp1, recLatMuVar, labLat, lossRec, lossKL, lossCls = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        mask = self.OTO.OTO.weight.detach()

        for _ in self.domains:
            inputsNew[_] = [Variable(inputs[_][0]).to(self.device), Variable(inputs[_][1]).to(self.device)]
            temp1[_] = self.OTO(inputsNew[_][0])
            recLatMuVar[_] = self.net[_](temp1[_])
            labLat[_], reduced = self.netCondClf(recLatMuVar[_][1])

            lossRec[_] = self.criterionRec(inputsNew[_][0] * mask, recLatMuVar[_][0]) / (inputsNew[_][0].shape[0] * (0.5))
            lossKL[_] = self.criterionKL(recLatMuVar[_][2], recLatMuVar[_][3], epoch, self.lambd)
            lossCls[_] = self.criterionCls(labLat[_], inputsNew[_][1])

        LossRec = self.alpha*torch.stack(list(lossRec.values()), dim=0).sum(dim=0).sum(dim=0) / len(self.domains)
        LossKL = torch.stack(list(lossKL.values()), dim=0).sum(dim=0).sum(dim=0) / len(self.domains)
        LossCls = self.beta*torch.stack(list(lossCls.values()), dim=0).sum(dim=0).sum(dim=0) / len(self.domains)
        LossSparse = self.gamma*torch.norm(self.OTO.OTO.weight, p=1)

        if self.classTrain and self.AETrain:
            Loss = LossCls + LossRec + LossKL
            if self.sparse:
                Loss += LossSparse

        Loss.backward()

        self.optOTO.step()
        for _ in self.domains:
            self.optNet[_].step()
        self.optNetCondClf.step()

        summary = dict()
        for _ in self.domains:
            # In fact we just write the average loss for both domains here
            # it can be updated to save each loss and recalculate average at the ouput
            summary[_] = dict()
            summary[_]['loss'] = Loss.item()
            summary[_]['lossCls'] = LossCls.item()
            summary[_]['lossRec'] = LossRec.item()
            summary[_]['lossKL'] = LossKL.item()
            summary[_]['lossSparse'] = LossSparse.item()

        return summary

    def runIter(self, inputs, epoch):
        self.OTO.eval()
        for _ in self.domains:
            self.net[_].eval()
        self.netCondClf.eval()
        inputsNew, temp1, recLatMuVar, labLat, lossRec, lossKL, lossCls = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        mask = self.OTO.OTO.weight.detach()
        for _ in self.domains:
            inputsNew[_] = [Variable(inputs[_][0]).to(self.device), Variable(inputs[_][1]).to(self.device)]
            temp1[_] = self.OTO(inputsNew[_][0])
            recLatMuVar[_] = self.net[_](temp1[_])
            labLat[_], reduced = self.netCondClf(recLatMuVar[_][1])

            lossRec[_] = self.criterionRec(inputsNew[_][0] * mask, recLatMuVar[_][0]) / (inputsNew[_][0].shape[0] * (0.5))
            lossKL[_] = self.criterionKL(recLatMuVar[_][2], recLatMuVar[_][3], epoch, self.lambd)
            lossCls[_] = self.criterionCls(labLat[_], inputsNew[_][1])

        LossRec = torch.stack(list(lossRec.values()), dim=0).sum(dim=0).sum(dim=0) / len(self.domains)
        LossKL = torch.stack(list(lossKL.values()), dim=0).sum(dim=0).sum(dim=0) / len(self.domains)
        LossCls = torch.stack(list(lossCls.values()), dim=0).sum(dim=0).sum(dim=0) / len(self.domains)
        LossSparse = self.gamma*torch.norm(self.OTO.OTO.weight, p=1)

        if self.classTrain:
            Loss = self.beta*LossCls
            if self.AETrain:
                Loss += self.alpha*LossRec + LossKL
                if self.sparse:
                    Loss += LossSparse
            else:
                if self.sparse:
                    Loss = LossSparse
        else:
            if self.AETrain:
                Loss = self.alpha*LossRec + LossKL
                if self.sparse:
                    Loss += LossSparse
            else:
                if self.sparse:
                    Loss = LossSparse

        summary = dict()
        for _ in self.domains:
            # In fact we just write the average loss for both domains here
            # it can be updated to save each loss and recalculate average at the ouput
            summary[_] = dict()
            summary[_]['loss'] = Loss.item()
            summary[_]['lossCls'] = LossCls.item()
            summary[_]['lossRec'] = LossRec.item()
            summary[_]['lossKL'] = LossKL.item()
            summary[_]['lossSparse'] = LossSparse.item()

        return summary

    def trainLoop(self, Train, Valid, nEpochs):
        losses = dict()
        lossesVal = dict()
        for _ in self.domains:
            losses[_] = dict()
            losses[_]['loss'], losses[_]['lossCls'], losses[_]['lossRec'], losses[_]['lossKL'], losses[_]['lossSparse'] = [], [], [], [], []
            lossesVal[_] = dict()
            lossesVal[_]['loss'], lossesVal[_]['lossCls'], lossesVal[_]['lossRec'], lossesVal[_]['lossKL'], lossesVal[_]['lossSparse'] = [], [], [], [], []
        for epoch in range(nEpochs):
            inputs = dict()
            inputsValid = dict()

            for _ in self.domains:
                inputs[_] = next(iter(Train[_]))
                inputsValid[_] = next(iter(Valid[_]))
            batchSize = max(inputs[_][0].shape[0] for _ in self.domains)
            for _ in self.domains:
                inputs[_] = inputs[_][0][:batchSize, :], inputs[_][1][:batchSize]
            summary = self.trainIter(inputs, epoch)
            summaryValid = self.runIter(inputsValid, epoch)


            for _1 in self.domains:

                for _2 in list(summary[_1].keys()):
                    losses[_1][_2].append(summary[_1][_2])
                for _2 in list(summary[_1].keys()):
                    lossesVal[_1][_2].append(summaryValid[_1][_2])
        
        return losses, lossesVal

    def getWeightsSparse(self):
        W = self.OTO.OTO.weight.detach().cpu().numpy()
        return W