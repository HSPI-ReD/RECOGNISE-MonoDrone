# configuraqtion
# check the MQTT broker IP and change if needed
# check the telegram apiToken, and chatID and change if needed

import base64
import numpy as np
import paho.mqtt.client as mqtt
import cv2, time
import pandas as pd
from ultralytics import YOLO
import requests
import torch
# import face_blur2
import winsound
from datetime import datetime
import pytz
#Deque is basically a double ended queue in python, we prefer deque over list when we need to perform insertion or pop 
# up operations at the same time
from collections import deque
import cvzone
import math
from sort import *

# print(class_list)
count=0

warningZone1 = [(117,147), (261,112), (275,131), (321,115), (637,367), (329,480)]

speedZone1 = [(120,151),(269,118),(321,154),(152,198)]
speedZone2 = [(240,327),(486,243),(540,288),(277,393)]

# anything inside warning zone will be in this set
warningZone_counter = set()

data_deque = {}

# get the standard UTC time
UTC = pytz.utc
# it will get the time zone
# of the specified location
IST = pytz.timezone('Europe/Rome')

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
frame_time = 0

speedLimit = 1 #SPEEDLIMIT
fpsFactor = 3 #TO COMPENSATE FOR SLOW PROCESSING
markGap = 1.8 #DISTANCE IN METRES BETWEEN THE MARKERS
startTracker = {} #STORE STARTING TIME OF PEOPLE
endTracker = {} #STORE ENDING TIME OF PEOPLE


def estimateSpeed(eID):
    timeDiff = endTracker[eID]-startTracker[eID]
                # NOT COPMPLETED YET +++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # ref. == vehicle-speed-detection file in yolov8 collection of RIPARTI project
    speed = round(markGap/timeDiff,2)
                # NOT COPMPLETED YET +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return speed


model=YOLO('yolov8m.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

MQTT_BROKER = "10.0.13.188"
MQTT_RECEIVE = "home/server"

frame_mqtt = np.zeros((240, 320, 3), np.uint8)

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_RECEIVE)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global frame_mqtt
    # Decoding the message
    img = base64.b64decode(msg.payload)
    # converting into numpy array from buffer
    npimg = np.frombuffer(img, dtype=np.uint8)
    # Decode to Original Frame
    frame_mqtt = cv2.imdecode(npimg, 1)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER)

# Starting thread which will receive the frames
client.loop_start()

my_file = open("coco.txt", "r")
classNames = my_file.read()
class_list = classNames.split("\n")


mask = cv2.imread('mask.png')


# Tracking
# age is the limit of the number of frames that we wait to deteck it back
# iou thr :: intersection over union threshold
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    
    # a frame scape method to speed up the algorithm. 
    # count += 1
    # if count % 3 != 0:
    #     continue
    map  = cv2.imread('map.png')
    
    frame1 = cv2.blur(frame_mqtt, (1,1)) 
    
    frame=cv2.resize(frame1, (640,480))
    
    
    
    # fps calculation ===============================================
    frame_time = time.time()
    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(frame_time-prev_frame_time)
    prev_frame_time = frame_time
    # converting the fps into integer
    fps = round(fps,2)
    print('fps is: ', fps, "\n" )  
    # fps calculation ===============================================
    
    
    
    imageRegion = cv2.bitwise_and(frame, mask)
    
    # results=model.predict(imageRegion, classes=[0,56])
    results=model.predict(frame, classes=[0,56])
    
    detections=np.empty((0, 5))
    
    # list of detected objects
    list=[]

    list_objectInZone=[]
    
    # W.Z.2 border
    cv2.line(frame, (210,415), (633,260), (0,165,255), 10)
    cv2.line(map, (210,415), (633,260), (0,165,255), 10)
    cv2.line(imageRegion, (210,415), (633,260), (0,165,255), 10)
    

    cv2.polylines(map, [np.array(speedZone1,np.int32)], True, (204,255,255), 2)
    cv2.polylines(map, [np.array(speedZone2,np.int32)], True, (204,255,255), 2)
    
    # draw warning zone1    
    cv2.polylines(map, [np.array(warningZone1,np.int32)], True, (0,165,255), 2)

    # *****************************************************
    # added in this version*******************************
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]     
            x1,y1,x2,y2 = int(x1),int(y1) ,int(x2) ,int(y2)   
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),1)
            w,h = x2-x1,y2-y1
            
            
            # find out confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = class_list[cls]
            if conf > 0.3 :
                # cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                #     scale=1, thickness=1)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=12)
                
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

        


    # *****************************************************
    # *****************************************************

    
    resultsTracker = tracker.update(detections)
    
    for result in resultsTracker: 
        
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1) ,int(x2) ,int(y2)  
        # print(result)
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=12)
        cvzone.putTextRect(frame, f'{currentClass} {int(id)} {conf}', (max(0, x1), max(35, y1)), 
            scale=1, thickness=1)
        
        cx=int(x1+x2)//2
        # cy=int(y3+y4)//2
        cy=y2
        center = (cx,cy)
        cv2.circle(frame, (cx,cy), radius=4, color=(0, 255, 255), thickness=-1)
        cv2.circle(imageRegion, (cx,cy), radius=10, color=(0, 255, 255), thickness=-1)
        # cv2.circle(map, (cx,cy), radius=10, color=(0, 255, 255), thickness=-1)
        # cv2.circle(map, (cx,cy), radius=20, color=(0, 255, 255), thickness=2)
        
        
        if cx >= ((-2.73 * cy) + 1342.8 ):
            cv2.line(frame, (210,415), (633,260), (0,0,255), 10)
            winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
            print('line alert')
            strID2 = str(id)
            print("IST time : ",  datetime.now(IST))
            
        Polygon_result = cv2.pointPolygonTest(np.array(warningZone1,np.int32), ((cx,cy)), False)
        if Polygon_result >= 0:
            list_objectInZone.append([cx])
                        
            warningZone_counter.add(id)
            
            cv2.circle(frame, (cx,cy), radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(imageRegion, (cx,cy), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.circle(map, (cx,cy), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.circle(map, (cx,cy), radius=20, color=(0, 0, 255), thickness=2)
            
            if id not in data_deque:  
                data_deque[id] = deque(maxlen= 64)
            # obj_name = class_list[int(id)]
            # label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)
            # add center to buffer
            data_deque[id].appendleft(center)
            # draw trail
            for i in range(1, len(data_deque[id])):
                # check if on buffer value is none
                if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                    continue
                # generate dynamic thickness of trails
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                # draw trails
                cv2.line(map, data_deque[id][i - 1], data_deque[id][i], (255,0,0), thickness)       
            
            
            
            
            # NOT COPMPLETED YET +++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Polygon_result_speed_1 = cv2.pointPolygonTest(np.array(speedZone1,np.int32), ((cx,cy)), False)
            if Polygon_result_speed_1 >= 0:
                startTracker[id] = time.time()
                print("start track is detected")
                
            Polygon_result_speed_2 = cv2.pointPolygonTest(np.array(speedZone2,np.int32), ((cx,cy)), False)
            if Polygon_result_speed_2 >= 0:
                endTracker[id] = time.time()
                print("end track is detected")            
                if startTracker:    
                    speed = estimateSpeed(id)

                    if speed > speedLimit:
                        print('Entity-ID :: {} = {} mps -- Overspeed is detected // THIS VALUE IS FAKE AND NOT FINALIZED IN THE CODE'.format(id, speed))
                    else:
                        print('Entity-ID : {} : {} mps // // THIS VALUE IS FAKE AND NOT FINALIZED IN THE CODE'.format(id, speed))
                        
                    startTracker.clear()
                    endTracker.clear()
                    
                    break
            # NOT COPMPLETED YET +++++++++++++++++++++++++++++++++++++++++++++++++++++++                
                  
            
            
            
    count1=len(list_objectInZone)
    copunt2=len(warningZone_counter)
    
    # this will print the id of the objects in the zone
    print('now is inside warning zone: ', count1)
    print('Entered in the warning zone: ', copunt2)      
    
    

     
    cv2.rectangle(frame, (0,455), (340, 480), (0,0,0), -1)
    cv2.putText(frame, 'N. of obj.s in the W.Zone: ' + str(count1),(6,475),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2 )

    cv2.imshow("RGB", frame)
    # cv2.imshow("ImageRegion", imageRegion)
    cv2.imshow("Map", map)

    # 1 in wait key ill start the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
    # in this mood the first frame will be shown and next frame will be appear after pressing SPACE botton
    # if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    

# Stop the Thread
client.loop_stop()    

cv2.destroyAllWindows()