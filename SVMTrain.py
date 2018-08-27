import cv2
import os
from sklearn.svm import LinearSVC
import numpy as np
import pickle
trainData = []
labels = []
err = 0
print('Reading files...')
count = 0
filesNum = len(os.listdir("..\\DATASET\\CHAR\\labeled"))
for f in os.listdir("..\\DATASET\\CHAR\\labeled"):
    errfounded = False
    count += 1
    if f.endswith(".jpg"):
        img = cv2.imread("../DATASET/CHAR/labeled/"+f)
        try:
            img = np.array(img.reshape((1, -1)))
        except:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img.reshape((1, -1)))
            except:
                err += 1
                errfounded = True
                print(f)
        if not errfounded:
            raw_label = f.split('-')[0]
            if raw_label.isdigit():
                try:
                    img_label = '0123456789-กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮาเ'[int(raw_label)]
                    labels.append(img_label)  
                    trainData.append(img)
                except:
                    print(f+" can't label!")
            else:
                img_label = '-'
                labels.append(img_label)  
                trainData.append(img)
        if count % 100 == 0:
            print("%.2f percent read." % (count/filesNum*100))
print("%d errors!" % err)
print('Images added to training set!')
trainData = np.array(trainData)
nsamples, nx, ny = trainData.shape
d2_train_dataset = trainData.reshape((nsamples,nx*ny))

print('Training...')
clf = LinearSVC()
clf.fit(d2_train_dataset, labels)
print('Trained')
pickle.dump(clf, open('OCRModel2.pkl','wb'))
