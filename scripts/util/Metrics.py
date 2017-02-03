'''
Created on 01.02.2017

@author: bastianbertram
'''
import subprocess as sub
import sys
import os

PATH = '../data/testResult/'


def eval_MAE(test_data, y_pred, PRED_FILE, TRUE_FILE):
    TRUE_FILE = os.path.join(PATH, TRUE_FILE)
    PRED_FILE = os.path.join(PATH, PRED_FILE)
    CALL = "perl Metrics.pl " + TRUE_FILE + " " + PRED_FILE
    
    pred_data = test_data.copy()
    del pred_data["tweet"]
    pred_data["yLabel"] = y_pred
    
    for i in range(0, len(pred_data["yLabel"])):
        if pred_data.loc[i, "yLabel"] == 4:
            pred_data.loc[i, "yLabel"]  = -1
        elif pred_data.loc[i, "yLabel"] == 3:
            pred_data.loc[i, "yLabel"] = -2
        
    print(pred_data)   
    
    pred_data.to_csv(PRED_FILE, header=None, index=None, sep='\t', mode='a')
    test_data.to_csv(TRUE_FILE, header=None, index=None, sep='\t', mode='a')
    
    sub.call(CALL, shell=True)
    

