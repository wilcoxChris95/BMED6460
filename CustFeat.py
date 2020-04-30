import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import trackpy as tpy

def Custom_DF(lblVid, origVideo, T=0):
    feats=pd.DataFrame()
    vidLength=len(lblVid)
    Frame_numObjs=[]

    for t in range(vidLength):
        lblFrame=lblVid[t]
        origFrame=origVideo[t]
        numObj=np.max(lblFrame)
        numFeats=0
        for k in range(1,numObj):
            #Create Binary array for objects of label k
            obj=lblFrame==k
            if(np.max(obj)>0):
                lblIndeces=np.where(obj==True)
                '''
                #Get Average Intensity Features
                '''
                N=(obj==1).sum()
                if(N>0):
                    avg_int=np.sum(origFrame[obj])/N
                else:
                    avg_int=0
                '''
                #Get Centroid Locations
                '''

                centroids=np.argwhere(obj==1).sum(0)/N
                x0=centroids[1]
                y0=centroids[0]

                '''
                #Used for calculating trajectories between 2 designated frames
                '''
                if(vidLength>1):
                    feats=feats.append([{'y': y0, 'x':x0, 'avgInt': avg_int, 'frame':t, 'size':N}])
                else:
                    feats=feats.append([{'y': y0, 'x':x0, 'avgInt': avg_int, 'frame':T, 'size':N}])
                numFeats+=1
            else:
                continue
        print("Custom Features: Frame "+str(t)+": "+str(numFeats)+" features")
    return(feats)