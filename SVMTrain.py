import cv2
import os
from sklearn.svm import LinearSVC
import numpy as np
import pickle
trainData = []
labels = []
print('Reading files...')
for f in os.listdir("..\\DATASET\\DIGITS"):
    img = cv2.imread('..\\DATASET\\DIGITS\\'+f)
    img_label = f[:1]
    img = np.array(img.reshape((1, -1)))
    trainData.append(img)
    labels.append(img_label)
trainData = np.array(trainData)
nsamples, nx, ny = trainData.shape
d2_train_dataset = trainData.reshape((nsamples,nx*ny))

print('Training...')
clf = LinearSVC()
clf.fit(d2_train_dataset, labels)
print('Trained')
pickle.dump(clf, open('OCRModel.pkl','wb'))
