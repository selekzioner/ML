import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

# 0 - true, 1 - false, gt - ground truth, th - threshold

def TP(probs, gt, th):
    return np.sum((probs > th) & (gt == 0))

def FP(probs, gt, th):
    return np.sum((probs > th) & (gt == 1))

def TN(probs, gt, th):
    return np.sum((probs <= th) & (gt == 1))

def FN(probs, gt, th):
    return np.sum((probs <= th) & (gt == 0))

def TPR(probs, gt, th):
    return TP(probs, gt, th) / (TP(probs, gt, th) + FN(probs, gt, th))

def FPR(probs, gt, th):
    return FP(probs, gt, th) / (TN(probs, gt, th) + FP(probs, gt, th))

def precision(probs, gt, th):
    return TP(probs, gt, th) / (TP(probs, gt, th) + FP(probs, gt, th))

def recall(probs, gt, th):
    return TPR(probs, gt, th)


def drawROC(probs, gt):
    ths = np.unique(probs.copy())
    ths.sort()
    tprs = [1]
    fprs = [1]
    for t in ths:
        if (TP(probs, gt, t) + FN(probs, gt, t) == 0 or TN(probs, gt, t) + FP(probs, gt, t) == 0):
            continue
        tprs.append(TPR(probs, gt, t))
        fprs.append(FPR(probs, gt, t))
    
    plt.plot(fprs, tprs)
    area = np.trapz(tprs, fprs)
    print('Area :' + str(-1 * area))
    
    fpr, tpr, _ = metrics.roc_curve(gt, probs)
    plt.plot(fpr, tpr)
    area = np.trapz(fpr, tpr)
    print('skilearn area :' + str(area))
    

def drawPR(probs, gt):
    ths = np.unique(probs.copy())
    ths.sort()
    rs = [1]
    ps = [0]
    for t in ths:
        if (TP(probs, gt, t) + FN(probs, gt, t) == 0 or TP(probs, gt, t) + FP(probs, gt, t) == 0):
            continue
        rs.append(recall(probs, gt, t))
        ps.append(precision(probs, gt, t))
        
    #rs.append(0)
    #ps.append(1)
    
    plt.plot(ps, rs)
    plt.xlim([0,1.1])
    plt.ylim([0,1.1])
    
    area = np.trapz(rs, ps)
    print('Area :' + str(area))
    
    