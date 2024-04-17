import numpy as np
from decimal import Decimal
import pickle
from filelock import FileLock
import matplotlib.pyplot as plt
from constants import *

def download_weights(DATA_NAME, EXPERIMENT):
    # global PATH_RESULTS, PATH_DATA
    
    if len(EXPERIMENT) >1:
        rootDir = PATH_RESULTS + DATA_NAME + 'all/'
    else:
        rootDir = PATH_RESULTS + DATA_NAME + EXPERIMENT[0] + '/'
    
    try:
        with FileLock(rootDir+"weightsList.pickle.lock"):
            weightsList = pickle.load( open(rootDir+"weightsList.pickle", "rb" ) )
            return weightsList
    except:
        print("There are no weights save for this experiment or domain, run the experiments first!")

def findThreshold(sortedW):
    X = np.arange(len(sortedW))
    P = np.hstack((X.reshape(-1,1),sortedW.reshape(-1,1)))
    m = (sortedW[0] - sortedW[-1])/(X[0] - X[-1])
    c = sortedW[0] - m * X[0]
    dists = []
    denom = np.sqrt(1+m**2)
    for p in P:
        x_0,y_0 = p[0],p[1]
        numerator = np.abs(m*x_0 + (-1)*y_0 + c)
        dists.append(numerator/denom)
    dists = np.array(dists)
    maxDistIndex = np.where(dists==max(dists))[0][0]
    return sortedW[maxDistIndex]


def plotElbow(DATA_NAME, EXPERIMENT, sortedW, title, Xtitle, Ytitle, name, scale, show):

    if len(EXPERIMENT) >1:
        rootDir = PATH_RESULTS + DATA_NAME + 'all/'
    else:
        rootDir = PATH_RESULTS + DATA_NAME + EXPERIMENT[0] + '/'

    X = np.arange(len(sortedW))
    P = np.hstack((X.reshape(-1,1),sortedW.reshape(-1,1)))
    m = (sortedW[0] - sortedW[-1])/(X[0] - X[-1])
    c = sortedW[0] - m * X[0]
    dists = []
    denom = np.sqrt(1+m**2)
    for p in P:
        x_0,y_0 = p[0],p[1]
        numerator = np.abs(m*x_0 + (-1)*y_0 + c)
        dists.append(numerator/denom)
    dists = np.array(dists)
    maxDistIndex = np.where(dists==max(dists))[0][0]

    fig = plt.figure(figsize=(8, 7))
    plt.plot(sortedW)
    plt.scatter(P[maxDistIndex,0],P[maxDistIndex,1],c='red')
    plt.text(maxDistIndex-150, max(sortedW) / 2, '# of sel. feat.=%s'%(maxDistIndex+1), fontsize = 34)
    plt.grid(False)
    # plt.axis('off')
    plt.title(title, size=38)
    plt.xlabel(Xtitle,size=38)
    plt.ylabel(Ytitle,size=38)
    plt.xticks(fontsize=33)
    plt.yticks(fontsize=33)
    # matplotlib.rcParams['font.size']=22
    if scale:
        # plt.rc('font', size=33)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText = True)
        # plt.rc('font', size=33)
    plt.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    plt.savefig(rootDir + name + '.png')
    if show:
        plt.show()
    else:
        plt.close()

    return maxDistIndex+1
    
     