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
    im = cv2.resize(image, (300, 120))
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im = cv2.equalizeHist(im)
    rects, show = findChar(im, False, True)
    text = ['','']
    rects = sorted(rects,key=lambda l:l[0])
    for i in range(len(rects)):
        cimg = imcrop(im, rects[i])
        cimg = cv2.resize(cimg, (12, 20))
        cimg = np.array(cimg.reshape((1, -1)))
        cimg = cv2.equalizeHist(cimg)
        c = skSVMclassifier.predict(cimg)
        if c[0] == '-':
            pass
        elif rects[i][1] > im.shape[0]*0.5:
            if rects != 0:
                if rects[i-1][0]+(rects[i-1][2]*2) >= rects[i][0]:
                    text[1] += ' '
            text[1] += c[0]
        else:
            text[0] += c[0]
        #cv2.putText(im,c[0],(rects[i][0],rects[i][1]), font, 1,(0,255,255),2,cv2.LINE_AA)
    if getRectangles:
        return (text[0], text[1], im, rects)
    else:
        return (text[0], text[1])


