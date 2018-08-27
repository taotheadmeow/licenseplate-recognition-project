import cv2
from sklearn.svm import LinearSVC
import numpy as np
import pickle
from charLocalization import findChar, imcrop
import time
font = cv2.FONT_HERSHEY_SIMPLEX

def licenseplateOCR(image, skSVMclassifier, getRectangles=False):
    """"
    This function will return 2 row of text those are license number and province
    """
    im = cv2.resize(image, (200, 80))
    rects, show = findChar(im, False, True)
    text = ['','']
    rects = sorted(rects,key=lambda l:l[0])
    for i in range(len(rects)):
        cimg = imcrop(im, rects[i])
        cimg = cv2.resize(cimg, (12, 20))
        cimg = np.array(cimg.reshape((1, -1)))
        c = skSVMclassifier.predict(cimg)
        if c[0] == '-':
            pass
        elif rects[i][1] > im.shape[0]*0.5:
            text[1] += c[0]
        else:
            text[0] += c[0]
        #cv2.putText(im,c[0],(rects[i][0],rects[i][1]), font, 1,(0,255,255),2,cv2.LINE_AA)
    if getRectangles:
        return (text[0], text[1], im)
    else:
        return (text[0], text[1])

clf = pickle.load(open('OCRModel.pkl','rb'))
im = cv2.imread('F:\\ALPR\\DATASET\\Test\\p0002.JPG')
t1, t2, im = licenseplateOCR(im, clf, True)
t = t1+","+t2
print(t)
cv2.imshow('w', im)
