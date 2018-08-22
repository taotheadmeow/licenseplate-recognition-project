import cv2
import numpy
import pytesseract
import os
import time

def findChar(img, show=False):
    """
    THIS WILL RETURN RECTANGLE IN LIST
    """
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)
    im_width, im_height = img.shape
    im_size = im_width*im_height
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.cv2.CHAIN_APPROX_NONE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]
    print (im_size)
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    charsLocation = []
    for rect in rects:
        # Draw the rectangles
        if rect[2]/rect[3] < 3 and rect[2]/rect[3] > 0.25 and rect[2]*rect[3] > im_size*0.001 and rect[2]*rect[3] < im_size*0.1:
            if show:
                cv2.rectangle(im2, (rect[0], rect[1]), (rect[0] + rect[2],\
                                                   rect[1] + rect[3]), (0, 255, 0), 1)
            charsLocation.append(rect)   
    if show:
        cv2.imshow('w', im2)
    return charsLocation

def imcrop(img, rectArray):
    return img[rectArray[1]:rectArray[1]+rectArray[3], rectArray[0]:rectArray[0]+rectArray[2]]

for f in os.listdir("..\\DATASET\\SignOnly"):
    img = cv2.imread('..\\DATASET\\SignOnly\\'+f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = findChar(img)
    for rect in rects:
        filename = str(hex(hash(time.time())))[2:]
        cv2.imwrite("..\\DATASET\\CHAR\\"+filename+".jpg", imcrop(img, rect))
        print(filename+" saved.")
