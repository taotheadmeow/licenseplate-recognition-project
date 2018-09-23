######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from ohSeeAh import licenseplateOCR
from sklearn.svm import LinearSVC
import pickle
from sectorPredict import predict_sector as ps
from sectorPredict import sector as provinces
from statistics import mode
import _thread
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from onvif_utils import getONVIFuri
#custom initialize
import time
import json
import mysql.connector
import socket
SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SOCKET.bind(("localhost", 6969))
SOCKET.listen(1)
IS_RUN = True

def stopLoop():
    global IS_RUN
    print('Shutting Down!')
    IS_RUN = False
    
def triggerListener(a,b):
    while(True):
        conn, addr = SOCKET.accept()
        data = conn.recv(32)
        conn.close()
        stopLoop()

_thread.start_new_thread( triggerListener, (None,None))

with open('appSettings.json', 'r') as infile:  
    configs = json.load(infile)
try:
    cnx = mysql.connector.connect(**configs['mysql'])
    cursor = cnx.cursor()
    cursor.execute("SELECT CAM_ID, CAM_URI, CAM_PORT, CAM_TYPE, CAM_USERNAME, CAM_PASSWORD FROM camera WHERE CAM_ACTIVE = 1;")
    result = cursor.fetchall()
    
except Exception as e:
    print("error: Can't connect to SQL server! Application aborting!:", e)
    exit()
    
MASTER_FRAME = {}
VIDEO_NAME = {}
video = {}
clf = pickle.load(open('OCRModel2.pkl','rb'))
MODE = 'online'
for row in result:
    if row[3] == "ONVI":
        VIDEO_NAME[row[0]] = getONVIFuri(row[1], row[4], row[5], row[2])
        print(VIDEO_NAME[row[0]], 'added') 
    elif row[3] == 'URI':
        try:
            if len(row[4])-row[4].count(' ')+len(row[5])-row[5].count(' ') != 0:
                VIDEO_NAME[row[0]] = (row[4]+':'+row[5]+'@'+row[1])
            else:
                VIDEO_NAME[row[0]] = (row[1])
        except:
            VIDEO_NAME[row[0]] = (row[1])
    MASTER_FRAME[row[0]] = np.zeros((200,200,3), dtype=np.uint8) #initial for frame storage
        
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

def startVideo(uri , stream_id):
    global MASTER_FRAME
    global ret
    global video
    video[stream_id] = cv2.VideoCapture(uri)
    while(video[stream_id].isOpened()):
        if not(IS_RUN):
            break
        else:
            ret, MASTER_FRAME[stream_id] = video[stream_id].read()
        

def runSession(detection_graph, sess, real_uri, stream_id):
    global IS_RUN
    global MASTER_FRAME

    ### CODE BELOW WILL GOES HERE

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize feed


    try:
        _thread.start_new_thread( startVideo, (real_uri, stream_id) )
    except Exception as e:
       print("Error: ", e, "Source ID:" , stream_id)
##    for i in range(5):
##        if type(MASTER_FRAME[stream_id]) == type(None):
##            print("Waiting for stream start... %d/5 sec." % i)
##            time.sleep(1)
##        else:
##            break
    try:
        im_height = MASTER_FRAME[stream_id].shape[0]
        im_width = MASTER_FRAME[stream_id].shape[1]
    except Exception as e:
        print("Error: unable to recieve video stream. Source ID:" , stream_id, e)
        exit()

    r = 0
    licenseplates = {'num':[], 'province':[]}
    latestLp = ''
    while(IS_RUN):
        if r == 0:
            licenseplates = {'num':[], 'province':[]}
        lpList = list()
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        try:
            frame = MASTER_FRAME[stream_id]
            timeStamp = int(time.time()*10)
            frame_expanded = np.expand_dims(frame, axis=0)

            # Perform the actual detection by running the model with the image as input
        
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
        except:
            print("Can't process video: Video streaming ended or camera disconnected! Source ID:", stream_id)
            print("Closing SQL Connection")
            video[row[0]].release()
            cnx.close()
            break

        # Extract result!
        
        for i in range(len(scores[0])):
            if scores[0][i] > 0.8:
                lpList.append(boxes[0][i])
                ## DRAW result
                cropped = frame[int(boxes[0][i][0]*im_height):int(boxes[0][i][2]*im_height),int(boxes[0][i][1]*im_width):int(boxes[0][i][3]*im_width)]
                cv2.rectangle(frame, (int(boxes[0][i][1]*im_width), int(boxes[0][i][0]*im_height)), (int(boxes[0][i][3]*im_width), int(boxes[0][i][2]*im_height)), (0,255,0), 1)
                if cropped.shape[0] < cropped.shape[1]:
                ##
                    text = licenseplateOCR(cropped,clf)
                    licenseplates['num'].append(text[0])
                    predicted_province, province_id = ps(text[1])[0][0], 1
                    # in the future!: predicted_province, provice_id = ps(text[1])[0][0]
                    licenseplates['province'].append(predicted_province)
                #print(text[0], predicted_province)
        r += 1
        if r >= 6 and len(licenseplates['num']) != 0 : 
            try:
                lpResult = mode(licenseplates['num'])
                if lpResult != latestLp and len(lpResult) > 2 and len(lpResult) <= 7 and len(lpResult)-sum(map(str.isdigit, lpResult)) < 3:
                    try:
                        cursor = cnx.cursor()
                        cursor.execute("""INSERT INTO licenseplate 
                                       (lp_id, lp_number, lp_datetime, lp_img_path, Camera_CAM_ID, Province_PROVINCE_ID) 
                                       VALUES (%s, %s, %s, %s, %s, %s)""", (timeStamp, lpResult, time.strftime('%Y-%m-%d %H:%M:%S'), configs['appconfig']['img_path'], stream_id, province_id))
                        #record Province will not work now
                        cnx.commit()
                        cursor.close()
                        cv2.imwrite(configs['appconfig']['img_path']+"\\"+str(stream_id)+str(timeStamp)+".jpg", \
                                    cropped)
                        cv2.imwrite(configs['appconfig']['img_path']+"\\"+str(stream_id)+str(timeStamp)+"f.jpg", \
                                    frame)
                        print("Recorded" , lpResult, mode(licenseplates['province']))
                    except Exception as e:
                        print(e)
                        print("Not Recorded" , lpResult, mode(licenseplates['province']))
                    latestLp = lpResult
                r = 0
            except:
                pass
        ##
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector '+str(stream_id), frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q') and not(IS_RUN):
            print("Closing SQL Connection")
            video[row[0]].release()
            cnx.close()
            break

for row in result:
    try:
        _thread.start_new_thread( runSession, (detection_graph, sess, VIDEO_NAME[row[0]], row[0]) )
        while(True):
            pass
            if not(IS_RUN):
                break
            
    except Exception as e:
        print(e)
        
# Clean up
cv2.destroyAllWindows()

