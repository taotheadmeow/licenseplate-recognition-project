######## Webcam Object Detection Using Tensorflow-trained Classifier #########

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
from statistics import mode, StatisticsError
import _thread
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from onvif_utils import getONVIFuri
#custom initialize
from multiprocessing import Pool
import time
import json
import mysql.connector
with open('appSettings.json', 'r') as infile:  
    configs = json.load(infile)
try:
    cnx = mysql.connector.connect(**configs['mysql'])
except Exception as e:
    print("error: Can't connect to SQL server! Application aborting!:", e)
    exit()

def insertLpToDict(d, lp_tup):
    if type(lp_tup) == type(None):
        pass
    elif lp_tup[0] in d:
        d[lp_tup[0]]['num'].append(lp_tup[1])
        d[lp_tup[0]]['province'].append(lp_tup[2])
    else:
        d[lp_tup[0]] = {'num':[lp_tup[1]], 'province':[lp_tup[2]]}
    return d
    

try:
    cursor = cnx.cursor()
    cursor.execute("SELECT CAM_ID, CAM_URI, CAM_PORT, CAM_TYPE, CAM_USERNAME, CAM_PASSWORD FROM camera WHERE CAM_ACTIVE = 1;")
    result = cursor.fetchall()
    if len(result) == 0:
        print("Camera not found! Add or activate camera to begin!")
        exit()
except Exception as e:
    print("Error table not found!")
    
MASTER_FRAME = {}
VIDEO_NAME = {}
#video = {}
clf = pickle.load(open('OCRModel2.pkl','rb'))
MODE = 'online'

def startVideo(uri , stream_id):
    print("Connecting to", uri)
    try:
        global MASTER_FRAME
        video = cv2.VideoCapture(uri)
        while(video.isOpened()):
            ret, MASTER_FRAME[stream_id] = video.read()
        video.release()
    except Exception as e:
        video.release()
        print(e)

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
            
    MASTER_FRAME[row[0]] = m = np.zeros((200,200,3), dtype=np.uint8) #initial for frame storage
        
def record(stream_id, licenseplates, r , cropped, frame, last_lp=''):
    global cnx
    global configs
    if r >= 7 and len(licenseplates['num']) != 0 : 
        try:
            lpResult = mode(licenseplates['num'])
            province = mode(licenseplates['province'])
            if lpResult != last_lp and len(lpResult) > 2 and len(lpResult) <= 7 and len(lpResult)-sum(map(str.isdigit, lpResult)) < 3:
                try:
                    cursor = cnx.cursor()
                    cursor.execute("""INSERT INTO licenseplate 
                                (lp_id, lp_number, lp_datetime, lp_img_path, Camera_CAM_ID, Province_PROVINCE_ID) 
                                VALUES (%s, %s, %s, %s, %s, %s)""", (timeStamp, lpResult, time.strftime('%Y-%m-%d %H:%M:%S'), configs['appconfig']['img_path'], stream_id, 1))
                    #record Province will not work now
                    cnx.commit()
                    cursor.close()
                    if type(cropped) is np.ndarray:
                        cv2.imwrite(configs['appconfig']['img_path']+"\\"+str(stream_id)+str(timeStamp)+".jpg", \
                                    cropped)
                        cv2.imwrite(configs['appconfig']['img_path']+"\\"+str(stream_id)+str(timeStamp)+"f.jpg", \
                                    frame)
                    print("Recorded" , lpResult, province)
                except Exception as e:
                    print(e)
                    print("Not Recorded" , lpResult, province)
                last_lp = lpResult
        except Exception as e:
            if type(e) is StatisticsError:
                pass
            else:
                print(e)
    return last_lp

def licenseplateDataProcess(frame_data):
    global configs
    text = [' ',' ']
    predicted_province=''
    boxes = frame_data['boxes']
    scores = frame_data['scores']
    classes = frame_data['classes']
    num = frame_data['num']
    frame = frame_data['frame']
    im_height = frame_data['frame'].shape[0]
    im_width = frame_data['frame'].shape[1] 
    for i in range(len(scores[0])):
        if scores[0][i] > 0.8:
            lpList.append(boxes[0][i])
            ## DRAW result
            cropped = frame[int(boxes[0][i][0]*im_height):int(boxes[0][i][2]*im_height),int(boxes[0][i][1]*im_width):int(boxes[0][i][3]*im_width)]
            if (cropped.shape[0] / cropped.shape[1]) < 0.8:
                cv2.rectangle(frame, (int(boxes[0][i][1]*im_width), int(boxes[0][i][0]*im_height)), (int(boxes[0][i][3]*im_width), int(boxes[0][i][2]*im_height)), (0,255,0), 1)
                
                text = licenseplateOCR(cropped,clf)
                #licenseplates['num'].append(text[0])
                predicted_province = ps(text[1])[0][0]
                #licenseplates['province'].append(predicted_province)
                cv2.imshow('Object detector '+str(frame_data['stream_id']), frame)
                print(text[0], predicted_province)
            return (frame_data['stream_id'], text[0], predicted_province, cropped, frame )
    cv2.imshow('Object detector '+str(frame_data['stream_id']), frame)



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

for row in result:
    try:
        _thread.start_new_thread( startVideo, (VIDEO_NAME[row[0]], row[0]) )
    except Exception as e:
        print("Error: ", e, "Source ID:" , stream_id)
# for i in range(5):
#     if type(MASTER_FRAME[stream_id]) == type(None):
#         print("Waiting for stream start... %d/5 sec." % i)
#         time.sleep(1)
#     elif i == 5:
#         print("Timeout!")
#         return
#     else:
#         break
# try:
#     im_height = MASTER_FRAME[stream_id].shape[0]
#     im_width = MASTER_FRAME[stream_id].shape[1]
# except Exception as e:
#     print("Error: unable to recieve video stream. Source ID:" , stream_id, e)
##     

r = 0
licenseplates = {}
last_lp = {}
frame_data = {}
cropped = {}
while(True):
    lpList = list()
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    # im_height = MASTER_FRAME[stream_id].shape[0]
    # im_width = MASTER_FRAME[stream_id].shape[1]
    timeStamp = int(time.time()*10)
    for i in MASTER_FRAME:
        frame = MASTER_FRAME[i]
        frame_expanded = np.expand_dims(frame, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        frame_data[i] = {'boxes':boxes, 'scores':scores, 'classes':classes, 'num':num, 'frame':frame, 'stream_id':i}
        #cv2.imshow('Object detector '+str(i), frame)

    # with Pool(os.cpu_count()) as p:
    #     processed_lp_datas = p.map(licenseplateDataProcess, frame_data.values())
    #     p.close()
    #     p.join()
    processed_lp_datas = map(licenseplateDataProcess, frame_data.values())
    for i in processed_lp_datas:
        #I is tuple (id, lp_num, province, cropped, frame)
        licenseplates = insertLpToDict(licenseplates, i)
        try:
            cropped[i[0]] = i[3]
            no_crop_im = False
        except:
            no_crop_im = True
    for i in licenseplates:
        if i in last_lp and no_crop_im:
            last_lp[i] = record(i, licenseplates.get(i), r, False, frame_data[i]['frame'], last_lp[i])
        elif i in last_lp and not no_crop_im:
            last_lp[i] = record(i, licenseplates.get(i), r, cropped[i], frame_data[i]['frame'], last_lp[i])
        elif i not in last_lp and not no_crop_im:
            last_lp[i] = record(i, licenseplates.get(i), r, cropped[i], frame_data[i]['frame'],)
        else:
            last_lp[i] = record(i, licenseplates.get(i), r, False, frame_data[i]['frame'])
    r+=1
    if cv2.waitKey(1) == ord('q'):
        print("Closing SQL Connection")
        cnx.close()
        #video[stream_id].release()
        break
    # All the results have been drawn on the frame, so it's time to display it.
    #cv2.imshow('Object detector '+'w', frame)

    # Press 'q' to quit
cv2.destroyAllWindows()