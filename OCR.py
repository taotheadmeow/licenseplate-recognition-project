import cv2
import numpy
import pytesseract

img = cv2.imread('..\\DATASET\\Test\\4k.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = cv2.bitwise_not(img)
im_width, im_height = img.shape
im_size = im_width*im_height
im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.cv2.CHAIN_APPROX_NONE)
rects = [cv2.boundingRect(ctr) for ctr in contours]
print (im_size)
im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
for rect in rects:
    # Draw the rectangles
    if rect[2]/rect[3] < 3 and rect[2]/rect[3] > 0.25 and rect[2]*rect[3] > im_size*0.001 and rect[2]*rect[3] < im_size*0.1:
        cv2.rectangle(im2, (rect[0], rect[1]), (rect[0] + rect[2],\
                                               rect[1] + rect[3]), (0, 255, 0), 1)
        print(rect[0], rect[1], rect[2]*rect[3])
    

#cv2.drawContours(im2, contours, -1, (0,255,0), 3)

cv2.imshow('w', im2)
print(pytesseract.image_to_string(cv2.bitwise_not(img), lang='myLang'))
# OR explicit beforehand converting
#print(pytesseract.image_to_string(Image.fromarray(img))

