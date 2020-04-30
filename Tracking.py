import matplotlib as mpl
import trackpy as tpy
import pandas
import numpy as np
from matplotlib.animation import FuncAnimation
import imageio
import os
import pims
import cv2
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
from AvgIntensity import *
from CustFeat import *
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from map import *
import motmetrics as mm
'''
DATA LOADING
'''


GT=[]
print("Gathering Gold Truth Data....")
#Dataset 5
list=os.listdir("DataSets/Training/PhC-C2Dl-PSC/PhC-C2DL-PSC/01_GT/TRA/")
#Dataset 4
#list=os.listdir("DataSets/Training/DIC-C2DH-Hela/RealData/DIC-C2DH-HeLa/02_GT/TRA/")
#Dataset 3
#list=os.listdir("DataSets/Training/DIC-C2DH-Hela/RealData/DIC-C2DH-HeLa/01_GT/TRA/")
#Dataset 2
#list=os.listdir("DataSets/Training/Fluo-N2DH-SIM+/02_GT/TRA")
#Dataset 1
#list=os.listdir("DataSets/Training/Fluo-N2DH-SIM+/01_GT/TRA")
for i in range(1,len(list)):
    #Dataset 5
    img=imageio.imread("DataSets/Training/PhC-C2Dl-PSC/PhC-C2DL-PSC/01_GT/TRA/"+list[i])
    #Dataset 4
    #img=imageio.imread("DataSets/Training/DIC-C2DH-Hela/RealData/DIC-C2DH-HeLa/02_GT/TRA/"+list[i])
    #Dataset 3
    #img=imageio.imread("DataSets/Training/DIC-C2DH-Hela/RealData/DIC-C2DH-HeLa/01_GT/TRA/"+list[i])
    #Dataset 2
    #img=imageio.imread("DataSets/Training/Fluo-N2DH-SIM+/02_GT/TRA/"+list[i])
    #Dataset 1
    #img=imageio.imread("DataSets/Training/Fluo-N2DH-SIM+/01_GT/TRA/"+list[i])
    GT.append(np.asarray(img))
print("Done Saving data.")
np.save('GT_TRACK.npy', GT)
print("Gold Truth has been saved to: GT_TRACK.npy")


print("Saving Segmented Images as Labelled and Binary Images...")
BinSeg_IMG=[]

#Dataset 5
list=os.listdir("DataSets/Training/PhC-C2Dl-PSC/01_RES/")
#Dataset 4
#list=os.listdir("DataSets/Training/DIC-C2DH-Hela/02_RES/")
#Dataset 3
#list=os.listdir("DataSets/Training/DIC-C2DH-Hela/01_RES/")
#Dataset 2
#list=os.listdir("DataSets/Training/Fluo-N2DH-SIM+/02_GT/SEG/")
#Dataset 1
#list=os.listdir("DataSets/Training/Fluo-N2DH-SIM+/01_GT/SEG/")

Seg_Img=[]

for i in range(len(list)):
    #Dataset 5
    img = imageio.imread("DataSets/Training/PhC-C2Dl-PSC/01_RES/" + list[i])
    #Dataset 4
    #img = imageio.imread("DataSets/Training/DIC-C2DH-Hela/02_RES/" + list[i])
    #Dataset 3
    #img = imageio.imread("DataSets/Training/DIC-C2DH-Hela/01_RES/" + list[i])
    #Dataset 2
    #img = imageio.imread("DataSets/Training/Fluo-N2DH-SIM+/02_GT/SEG/" + list[i])
    #Dataset 1
    #img = imageio.imread("DataSets/Training/Fluo-N2DH-SIM+/01_GT/SEG/" + list[i])
    Seg_Img.append(img)
    img=img>0
    BinSeg_IMG.append(img)

np.save('Bin_SegImgs.npy', BinSeg_IMG)
np.save('Seg_Img.npy', Seg_Img)
print("Binary Segmented Images have been saved to: Bin_SegImgs.npy and non-binary images have been saved to Seg_Img.npy")


print("Loading Gold Truth Track data...")
GT_TRACK=np.load('GT_TRACK.npy')
print("GT_Track.npy has been loaded, it has length: ", len(GT_TRACK))

print("Loading Binary Segmented Images...")
BinSeg_IMG=np.load('Bin_SegImgs.npy')
print("Bin_SegImgs.npy has been loaded, it has length: ", len(BinSeg_IMG))

print("Loading Non-Binary Segmented Images...")
Seg_Img=np.load('Seg_Img.npy')
print("Loading Ground Truth Data...")


#Takes argument t=frame number, Di=Detection
def makeDataFrame(t, Di):
    x_p=[]
    y_p=[]
    Area=[]
    for i in range(len(Di)):
        x_p.append(Di[i][0][1])
        y_p.append(Di[i][0][0])
        Area.append(Di[i][1])

    return pandas.DataFrame(dict(x=x_p, y=y_p, frame=t, area=Area))

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Graphs elements in each frame
def trshow(tr, first_style='bo', last_style='gs', style='b.'):
    frames = list(tr.groupby('frame'))
    nframes = len(frames)
    for i, (fnum, pts) in enumerate(frames):
        if i == 0:
            sty = first_style
        elif i == nframes - 1:
            sty = last_style
        else:
            sty = style
        print(pts.x, pts.y)
    tpy.plot_traj(tr, colorby='frame', ax=gca())
    axis('equal')
    xlabel('x')
    ylabel('y')

#Calculate Diameter
def located_D(l,w):
    if(l!=w):
        if np.floor(np.max([l,w]))%2 == 0:
            arg1=np.max([l,w])+1
        else:
            arg1=np.max([l,w])

        if np.floor(np.min([l,w]))%2 == 0:
            arg2=np.min([l,w])+1
        else:
            arg2=np.min([l,w])

        return np.floor([arg1, arg2, (1/2)*(np.max([l,w])-np.min([l,w]))])
    else:
        return np.floor([w, (1/2)*w])

#Dataset 5
frames = pims.ImageSequence('DataSets/Training/PhC-C2Dl-PSC/PhC-C2Dl-PSC/01/*.tif', as_grey=True)
#Dataset 4
#frames = pims.ImageSequence('DataSets/Training/DIC-C2DH-Hela/RealData/DIC-C2DH-HeLa/02/*.tif', as_grey=True)
#Dataset 3
#frames = pims.ImageSequence('DataSets/Training/DIC-C2DH-Hela/RealData/DIC-C2DH-HeLa/01/*.tif', as_grey=True)
#Dataset 2
#frames = pims.ImageSequence('DataSets/Training/Fluo-N2DH-SIM+/02/*.tif', as_grey=True)
#Dataset 1
#frames = pims.ImageSequence('DataSets/Training/Fluo-N2DH-SIM+/01/*.tif', as_grey=True)


print("Video Information: \n", frames)
numFiles=len(frames)
avg_Dims=[]
avg_Len = []
avg_Wid = []
numCells=[]

'''
Calculates the width and height of every object in frame using opencv
Created by Adrian Rosebrock.  Accessed from https://www.pyimagesearch.com/
'''
for i in range(numFiles):
    img=np.uint8(BinSeg_IMG[i])
    contours,hierarchy = cv2.findContours(img, 1, 2)
    Len = []
    Wid = []
    for c in contours:
        box=cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box=np.array(box, dtype="int")
        box=perspective.order_points(box)
        cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them

        for (x, y) in box:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 2)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 2)
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        Len.append(dA)
        Wid.append(dB)
        # draw the object sizes on the image
        # show the output image
    x_avg=np.sum(Len)/len(contours)
    y_avg=np.sum(Wid)/len(contours)
    avg_Len.append(x_avg)
    avg_Wid.append(y_avg)


print("Calculating the Average Intensity Mask...")
#AvgVid=AvgIntensityMask(Seg_Img[0:numFiles], frames, AvgMask=False, blur=True, edge_Detection=True)
print("Average intensity Mask of Images complete")
d=np.max(located_D(np.max(avg_Len),np.min(avg_Wid)))



'''
# Trackpy locate Features Demonstration
'''
f0_new=tpy.locate(frames[0],diameter =d, minmass=105, maxsize=None, separation=15, noise_size=1, smoothing_size=None,
threshold=None, invert=False, topn=30, preprocess= True, max_iterations=10, filter_before=None, filter_after=True,
characterize=False)

numFiles=int(numFiles*0.3)


#Create Custom Features Dataframe
print("Creating Custom Features...")
features=Custom_DF(Seg_Img[0:numFiles], frames)
print("Custom Features have been calculated")

print("Saving Custom Feaures to CDF.pkl...")
#features.to_pickle("CDF.pkl")
#features.to_pickle("CDF_DS3.pkl")

print("Loading Custom Features from CDF.pkl...")
#features=pandas.read_pickle("CDF.pkl")
#features=pandas.read_pickle("CDF_DS3.pkl")
print("Loading CDF.pkl done")
'''
feats=tpy.batch(frames, diameter=d, minmass=None, maxsize=59, noise_size=1, smoothing_size=None,
threshold=None, invert=False, topn=None, preprocess= True, max_iterations=10, filter_after=True,
characterize=True, engine='python', meta='Results/Batch_Info.txt')
'''

tr=tpy.link_df(features, search_range=10, adaptive_stop=3, adaptive_step=0.95, memory=3)
#tr.to_pickle("DS5tracks.pkl")
#tr=pandas.read_pickle("DS5tracks.pkl")
tr1=tpy.filter_stubs(tr, 0.1*numFiles)
drift=tpy.compute_drift(tr1)
#tr1=tr                                                     #No Truncation
trF=tpy.subtract_drift(tr1, drift)
#plt.show()
plt.clf()

'''
tempTr=trF
R=60
pairs=[]
for t in range(numFiles-1):
    frame0=trF.loc[[t], ['particle', 'x', 'y']]
    frame1=trF.loc[[t+1], ['particle', 'x', 'y']]
    nums0=frame0['particle']
    nums1=frame1['particle']
    for idx, rows in frame1.iterrows():
        num=rows['particle']
        parent_found=False
        if num not in nums0.values:
            #Get point of child cell
            x1=rows['x']
            y1=rows['y']
            prev_frame=tempTr.loc[[t],['particle', 'x','y']]
            min_radius=R
            for idx0, rows0 in prev_frame.iterrows():
                x_prev=rows0['x']
                y_prev=rows0['y']
                num_prev=rows0['particle']
                if np.sqrt((x1-x_prev)**2 +(y1-y_prev)**2)<=min_radius:
                    parent_found=True
                    parent_num=num_prev
                    pairs.append((parent_num,num))
                    min_radius=np.sqrt((x1-x_prev)**2 +(y1-y_prev)**2)
                if parent_found==True:
                    break
                else:
                    continue
'''

'''
#Show As Movie

'''

'''
# Optionally, tweak styles.
mpl.rc('figure', figsize=(10, 5))
mpl.rc('image', cmap='gray')

def cvtFig2Numpy(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height.astype(np.uint32),
                                                                        width.astype(np.uint32), 3)
    return image

def makevideoFromArray(movieName, array, fps=25):
    imageio.mimwrite(movieName, array, fps=fps);



arr = []
for i in range(numFiles):
    frame = frames[i]
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(frame)
    axes = tpy.plot_traj(tr1.query('frame<={0}'.format(i)))
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])
    arr.append(cvtFig2Numpy(fig))
    plt.close('all')

makevideoFromArray("Results3.mp4", arr, 5)
'''

'''
#Metrics
'''
print("Calculating Maps...")
GT_Map=MakeGTTrackMap(Seg_Img)
Calc_Map=MakeCalcTrackMap(trF, numFiles)

print("length of GT Map: ", len(GT_Map), " length of Calc Map: ", len(Calc_Map))
print("Saving Custom Feaures to CDF.pkl...")
#GT_Map.to_pickle("GT_Map.pkl")
#Calc_Map.to_pickle("Calc_Map.pkl")

print("Loading Custom Features from CDF.pkl...")
#GT_Map=pandas.read_pickle("GT_Map.pkl")
#Calc_Map=pandas.read_pickle("Calc_Map.pkl")

acc = mm.MOTAccumulator(auto_id=True)

print("Calculating Metrics...")
for t in range(numFiles):
    GT_frame=GT_Map.get(t)
    Calc_frame=Calc_Map.get(t)
    GT_lbls=[]
    Calc_lbls=[]
    for obj in GT_frame:
        GT_lbl=obj.get('label')
        GT_x=obj.get('x')
        GT_y=obj.get('y')
        GT_lbls.append(GT_lbl)
    for calc_Obj in Calc_frame:
        Calc_lbl=obj.get('label')
        calc_x=obj.get('x')
        calc_y=obj.get('y')
        Calc_lbls.append(Calc_lbl)
    GT_pts=[]
    Calc_pts=[]
    distances=[]
    for GT_obj in GT_frame:
        x_GT=GT_obj.get('x')
        y_GT=GT_obj.get('y')
        GT_pts.append(np.asarray([x_GT, y_GT]))
        norms=[]
    for calc_Obj in Calc_frame:
        x_Calc = calc_Obj.get('x')
        y_Calc = calc_Obj.get('y')
        norm=norm2([x_GT, x_Calc], [y_GT,y_Calc])
        Calc_pts.append(np.asarray([x_Calc, y_Calc]))
        norms.append(norm)
        '''
        for calc_Obj in Calc_frame:
            x_Calc = calc_Obj.get('x')
            y_Calc = calc_Obj.get('y')
            norm=norm2([x_GT, x_Calc], [y_GT,y_Calc])
            norms.append(norm)
        '''
    C = mm.distances.norm2squared_matrix(GT_pts, Calc_pts)
    #distances.append(norms)
    #print(distances)
    #print(len(distances))
    acc.update(GT_lbls, Calc_lbls, C)
    print("Frame"+str(t)+" complete")
    #print(acc.mot_events)
print("Metric Calculation Complete.")

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
strsummary=mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
print(strsummary)
