import numpy as np
import matplotlib.pyplot as plt
import cv2

def AvgIntensityMask(LabelledVideo, origVideo, blur=False, edge_Detection=False,AvgMask=True):
    newVideo=[]
    for t in range(len(LabelledVideo)):
        numObj=np.max(LabelledVideo[t])
        Lblframe = LabelledVideo[t]
        origFrame=origVideo[t]
        m,n = origFrame.shape
        newFrame=np.zeros((m,n))
        hor_shift=14
        ver_shift=16
        if AvgMask==False and edge_Detection==False:
            newFrame=origFrame
        if edge_Detection==True and AvgMask==True:
            newFrame=Lblframe
        for k in range(1,numObj):
            obj=Lblframe==k
            if np.max(obj)>0:
                lblIndeces=np.where(obj==1)
                N=len(lblIndeces[0])
                newFrame[lblIndeces]=np.sum(origFrame[lblIndeces])/N
            else:
                continue
        if (blur):
            newFrame = cv2.GaussianBlur(newFrame, (5, 5), sigmaX=3)
        if (edge_Detection):
            newFrame = cv2.Canny(np.uint8(newFrame), 100.0, 220.0, apertureSize=3, L2gradient=True)

        newVideo.append(newFrame)

    return(newVideo)
