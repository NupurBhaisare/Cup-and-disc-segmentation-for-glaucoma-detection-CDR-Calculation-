from scipy import signal
import cv2
import sys
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
from PIL import Image
import xlrd 
import math
from pylab import*
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))

#IMPORT GROUND TRUTH
#Provide the folder link to the ground truth file is the drishti dataset
wb = xlrd.open_workbook("E:/optic-cup-disc/Drishti-GS1_files/Drishti-GS1_diagnosis.xlsx") 
sheet = wb.sheet_by_index(0) 
val = [sheet.col_values(1)[5:],sheet.col_values(8)[5:]]

def load_image(path):
    # returns an image of dtype int in range [0, 255]
    return np.asarray(Image.open(path))

#function to load image and their name
def load_set(folder, shuffle=False):
    img_list = sorted(glob.glob(os.path.join(folder, '*.png')) + \
                      glob.glob(os.path.join(folder, '*.jpg')) + \
                      glob.glob(os.path.join(folder, '*.jpeg')))
    if shuffle:
        np.random.shuffle(img_list)
    data = []
    filenames = []
    for img_fn in img_list:
        img = load_image(img_fn)
        data.append(img)
        filenames.append(img_fn)
    return data, filenames


#DATA EXTRACTION FUNCTION
#db_folder = drishti dataset folder
#cdr = set it true to get the cdr values of 4 experts
#train_data = setting it true gives training data and false gives testing data
def extract_DRISHTI_GS_train(db_folder,cdr,train_data):

    file_codes_all,exp1,exp2,exp3,exp4 = [], [], [], [], []
    if train_data:
        set_path = os.path.join(db_folder, 'Drishti-GS1_files','Drishti-GS1_files', 'Training')
    else:
        set_path = os.path.join(db_folder, 'Drishti-GS1_files','Drishti-GS1_files', 'Test')
    images_path = os.path.join(set_path, 'Images')
    X_all, file_names = load_set(images_path)
    rel_file_names = [os.path.split(fn)[-1] for fn in file_names]
    rel_file_names_wo_ext = [fn[:fn.rfind('.')] for fn in rel_file_names]
    if train_data:
        file_codes = [fn[fn.find('_'):] for fn in rel_file_names_wo_ext]
    else:
        file_codes = [fn[fn.find('_'):] for fn in rel_file_names_wo_ext]
    file_codes_all.extend(file_codes)
    
    for fn in rel_file_names_wo_ext:
        if cdr:
            if train_data:
                CDR = open(os.path.join(set_path, 'GT', fn,fn + '_cdrValues.txt'),'r')
            else:
                CDR = open(os.path.join(set_path, 'Test_GT', fn,fn + '_cdrValues.txt'),'r')
            CDR = list(CDR)
            CDR = CDR[0].split()
            exp1.append(CDR[0])
            exp2.append(CDR[1])
            exp3.append(CDR[2])
            exp4.append(CDR[3])
            
    return X_all, file_codes_all,exp1,exp2,exp3,exp4,file_names
    #This functions returns the data images,their names and the corresponding cdr values of each expert in order


# GET DATA
X_all,file_codes_all,exp1,exp2,exp3,exp4,file_names = extract_DRISHTI_GS_train('E:/optic-cup-disc',True,False) #put the folder where dataset is


# FUNCTION TO SEGMENT CUP AND DISK
#image = fundus image
#plot_seg = plots the segmented image
#plt_hist = plots the histogram of red and green channel before and after smoothing
def segment(image,plot_seg,plot_hist):

    image = image[400:1400,500:1600,:] #cropping the fundus image to ger region of interest

    Abo,Ago,Aro = cv2.split(image)  #splitting into 3 channels
    #Aro = clahe.apply(Aro)
    Ago = clahe.apply(Ago)
    M = 60    #filter size
    filter = signal.gaussian(M, std=6) #Gaussian Window
    filter=filter/sum(filter)
    STDf = filter.std()  #It'standard deviation
    

    Ar = Aro - Aro.mean() - Aro.std() #Preprocessing Red
    
    Mr = Ar.mean()                           #Mean of preprocessed red
    SDr = Ar.std()                           #SD of preprocessed red
    Thr = 0.5*M - STDf - Ar.std()            #Optic disc Threshold
    #print(Thr)

    Ag = Ago - Ago.mean() - Ago.std()		 #Preprocessing Green
    Mg = Ag.mean()                           #Mean of preprocessed green
    SDg = Ag.std()                           #SD of preprocessed green
    Thg = 0.5*Mg +2*STDf + 2*SDg + Mg        #Optic Cup Threshold
    #print(Thg)
    
    
    hist,bins = np.histogram(Ag.ravel(),256,[0,256])   #Histogram of preprocessed green channel
    histr,binsr = np.histogram(Ar.ravel(),256,[0,256]) #Histogram of preprocessed red channel


    smooth_hist_g=np.convolve(filter,hist)  #Histogram Smoothing Green
    smooth_hist_r=np.convolve(filter,histr) #Histogram Smoothing Red
    
    #plot histogram if input is true
    if plot_hist:
        plt.subplot(2, 2, 1)
        plt.plot(hist)
        plt.title("Preprocessed Green Channel")

        plt.subplot(2, 2, 2)
        plt.plot(smooth_hist_g)
        plt.title("Smoothed Histogram Green Channel")

        plt.subplot(2, 2, 3)
        plt.plot(histr)
        plt.title("Preprocessed Red Channel")

        plt.subplot(2, 2, 4)
        plt.plot(smooth_hist_r)
        plt.title("Smoothed Histogram Red Channel")

        plt.show()
    
    r,c = Ag.shape
    Dd = np.zeros(shape=(r,c)) #Segmented disc image initialization
    Dc = np.zeros(shape=(r,c)) #Segmented cup image initialization

    #Using obtained threshold for thresholding of the fundus image
    for i in range(1,r):
        for j in range(1,c):
            if Ar[i,j]>Thr:
                Dd[i,j]=255
            else:
                Dd[i,j]=0

    for i in range(1,r):
        for j in range(1,c):
        
            if Ag[i,j]>Thg:
                Dc[i,j]=1
            else:
                Dc[i,j]=0
         
    #Saving the segmented image in the same place as the code folder      
    cv2.imwrite('disk.png',Dd)
    plt.imsave('cup.png',Dc)
    
    if plot_seg:
        plt.imshow(Dd, cmap = 'gray', interpolation = 'bicubic')
        plt.axis("off")
        plt.title("Optic Disk")
        plt.show()
        
        plt.imshow(Dc, cmap = 'gray', interpolation = 'bicubic')
        plt.axis("off")
        plt.title("Optic Cup")
        plt.show()



# FUNCTION TO CALCULATE CDR

#import cv2 as cv

def cdr(cup,disc,plot):
    
    #morphological closing and opening operations
    R1 = cv2.morphologyEx(cup, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)	
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)

    img = clahe.apply(r3)
    
    
    ret,thresh = cv2.threshold(cup,127,255,0)
    img,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Getting all possible contours in the segmented image
    cup_diameter = 0
    largest_area = 0
    el_cup = contours[0]
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >= 5:
                area = cv2.contourArea(contours[i]) #Getting the contour with the largest area
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_cup = cv2.fitEllipse(contours[i])
                
    cv2.ellipse(img,el_cup,(140,60,150),3)  #fitting ellipse with the largest area
    x,y,w,h = cv2.boundingRect(contours[index]) #fitting a rectangle on the ellipse to get the length of major axis
    cup_diameter = max(w,h) #major axis is the diameter

    #morphological closing and opening operations
    R1 = cv2.morphologyEx(disc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations = 1)
    r1 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
    R2 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,21)), iterations = 1)
    r2 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,1)), iterations = 1)
    R3 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)), iterations = 1)
    r3 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(43,43)), iterations = 1)

    img2 = clahe.apply(r3)
    
    ret,thresh = cv2.threshold(disc,127,255,0)
    img2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Getting all possible contours in the segmented image
    disk_diameter = 0
    largest_area = 0
    el_disc = el_cup
    if len(contours) != 0:
          for i in range(len(contours)):
            if len(contours[i]) >= 5:
                area = cv2.contourArea(contours[i]) #Getting the contour with the largest area
                if (area>largest_area):
                    largest_area=area
                    index = i
                    el_disc = cv2.fitEllipse(contours[i])
                    
    cv2.ellipse(img2,el_disc,(140,60,150),3) #fitting ellipse with the largest area
    x,y,w,h = cv2.boundingRect(contours[index]) #fitting a rectangle on the ellipse to get the length of major axis
    disk_diameter = max(w,h) #major axis is the diameter
                
    if plot:
        plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
        plt.axis("off")
        plt.title("Optic Disk")
        plt.show()
        plt.imshow(img)
        plt.axis("off")
        plt.title("Optic Cup")
        plt.show()
        
    if(disk_diameter == 0): return 1 # if disc not segmented properly then cdr might be infinity
    cdr = cup_diameter/disk_diameter #ration of major axis of cup and disc
    return cdr



# MAIN FUNCTION
CDR = [] # load calculated cdr here
VAL = [] # load their labels here
count = 0
for i in range(len(X_all)):
    set_path = os.path.join('E:/optic-cup-disc', 'Drishti-GS1_files','Drishti-GS1_files', 'Test',file_names[i]) #folder to the test image in dataset(don't change file_names[])
    image = cv2.imread(set_path,1)
    segment(image,False,False)
    cup = cv2.imread('E:/Term Project 01/cup.png',0) #images will be saved in the same folder as the code so that folder needs to be put here
    disc = cv2.imread('E:/Term Project 01/disk.png',0) #images will be saved in the same folder as the code so that folder needs to be put here
    cdr_cal = cdr(cup,disc,False)
    if(val[1][int(file_codes_all[count][1:])-1] == 'Glaucomatous'):
        VAL.append(1)
    else:
        VAL.append(0)
    CDR.append(cdr_cal)
    print(file_codes_all[count],'Exp1_cdr:',exp1[count],'Exp2_cdr:',exp2[count],'Exp3_cdr:',exp3[count],'Exp4_cdr:',exp4[count],'Pred_cdr:',cdr_cal)
    os.remove('E:/Term Project 01/cup.png') #put same folder as that of cup and disc so that they can be removed for processing of next set
    os.remove('E:/Term Project 01/disk.png')
    count+=1

error1,error2,error3,error4 = [],[],[],[]
#calculated error against each expert
for i in range(len(X_all)):
    error1.append(float(exp1[i]) - CDR[i])
    error2.append(float(exp2[i]) - CDR[i])
    error3.append(float(exp3[i]) - CDR[i])
    error4.append(float(exp4[i]) - CDR[i])

#saving the error and calculated cdr and its label into csv files for training classification model
a = pd.DataFrame(error1) #exper1
b = pd.DataFrame(error2) #exper2
c = pd.DataFrame(error3) #exper3
d = pd.DataFrame(error4) #exper4
x1 = pd.DataFrame(CDR) #calculated cdr
y1 = pd.DataFrame(VAL) #It's actual label
x1.to_csv('E:/Term Project 01/x1.csv',index=False)
y1.to_csv('E:/Term Project 01/y1.csv',index=False)
a.to_csv('E:/Term Project 01/a.csv',index=False)
b.to_csv('E:/Term Project 01/b.csv',index=False)
c.to_csv('E:/Term Project 01/c.csv',index=False)
d.to_csv('E:/Term Project 01/d.csv',index=False)


# CLASSIFICATION MODEL

#load all the required files
X_train = pd.read_csv('E:/Term Project 01/x.csv')
Y_train = pd.read_csv('E:/Term Project 01/y.csv')
X_test = pd.read_csv('E:/Term Project 01/x1.csv')
Y_test = pd.read_csv('E:/Term Project 01/y1.csv')
a = pd.read_csv('E:/Term Project 01/a.csv')
b = pd.read_csv('E:/Term Project 01/b.csv')
c = pd.read_csv('E:/Term Project 01/c.csv')
d = pd.read_csv('E:/Term Project 01/d.csv')

logreg = LogisticRegression() #train model with logistic regression

logreg.fit(X_train, Y_train)

y_pred = logreg.predict(X_test)

acc = f1_score(Y_test, y_pred) #f1score for the classification
print('Accuracy:',acc)
print('Mean Error Expert1:',np.mean(a)[0],' ','STD Error Expert1:',np.std(a)[0])
print('Mean Error Expert2:',np.mean(b)[0],' ','STD Error Expert2:',np.std(b)[0])
print('Mean Error Expert3:',np.mean(c)[0],' ','STD Error Expert3:',np.std(c)[0])
print('Mean Error Expert4:',np.mean(d)[0],' ','STD Error Expert4:',np.std(d)[0])