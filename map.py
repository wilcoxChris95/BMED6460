import numpy as np
import operator
import motmetrics as mm

def norm2(x,y):
    return(np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2))

def MakeGTTrackMap(GT_tr):
    M={}
    numFrames=len(GT_tr)
    for t in range(numFrames):
        numObj=np.max(GT_tr[t])
        for k in range(numObj):
            obj = GT_tr[t] == k
            if(np.max(obj)>0):
                lblIndeces = np.where(obj == True)
                N = (obj == 1).sum()
                centroids = np.argwhere(obj == 1).sum(0) / N
                x0 = centroids[1]
                y0 = centroids[0]
                if t in M:
                    M[t].append({'x':x0, 'y':y0, 'label':k})
                else:
                    M[t]=[{'x':x0,'y':y0, 'label':k}]
    return(M)

def MakeCalcTrackMap(tr, numFrames):
    M={}
    for t in range(numFrames):
        frame_Objs=tr.loc[[t], ['particle', 'x','y']]
        for idx, rows in frame_Objs.iterrows():
            num=rows['particle']
            x0=rows['x']
            y0=rows['y']

            if t in M:
                M[t].append({'x':x0,'y': y0, 'label':num})
            else:
                M[t]=[{'x': x0,'y': y0,'label': num}]
    return(M)
