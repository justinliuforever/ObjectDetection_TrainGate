import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math

from ROI_Select import ROI_SelectArea
model = torch.hub.load('ultralytics/yolov5', 'custom', path='trainresult/last_allGate_onlyorigin_0223.pt', force_reload=True)
model.conf = 0.1
videofile = "test_im&Vde/6255092968981447d491ef3a_1671631820.mp4"
videofile = "test_im&Vde/Peek62ed249f02fcc15ffec0ba49_1672577344.mp4"




# get infor from "Result", we can get bounding box value, name, ... in this function
def grabInf(results):
    Inf = results.pandas().xyxy[0]
    boxXY, boxName, boxConfi = [], [], []
    for index, row in Inf.iterrows():
        xyMin, xyMax = (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax']))
        boxXY.append((xyMin, xyMax))
        boxName.append(row['name'])
        boxConfi.append(row['confidence'])  
    #print(Inf)
    #print(boxName)
    return boxXY, boxName, boxConfi

# Using the highest confidence detection to calculate angle
def getAngle(objectDetBoxs):
    if objectDetBoxs == [] or objectDetBoxs ==  [[]]:
        return []
    # name, confidence,xtopletf, ytopleft, xdownright, ydownright
    i, length = 0, len(objectDetBoxs)
    highConf, angleDegree = 0, 0

    while i < length:
        objectDetBox = objectDetBoxs[i]
        objName, objConf,xMin, yMin, xMax, yMax = objectDetBox[0], objectDetBox[1], objectDetBox[2], objectDetBox[3], objectDetBox[4], objectDetBox[5]
        if objConf > highConf:
            highConf = objConf
            height = yMax - yMin
            distance = xMax - xMin
            angleDegree = math.degrees(math.atan(height/distance))
        i = i + 1
    return highConf, angleDegree

# Using the angle to determine OPEN or CLOSE
def checkStatus(gateAngles, judgeAngle):
    num_close = 0
    num_open = 0
    conclu = ["Close", "Open", "Empty"]

    for angle in gateAngles:
        if angle < judgeAngle:
            num_close += 1
        else:
            num_open += 1
    if num_close == 0 and num_open == 0:
        return conclu[2], num_close, num_open
    elif num_close == 0 and num_open != 0:
        return conclu[1], num_close, num_open
    else:
        return conclu[0], num_close, num_open


# ROI Multiple 
rois = ROI_SelectArea(videofile)
cap = cv2.VideoCapture(videofile)

while cap.isOpened():
    ret, frame = cap.read()
    objectDetBoxs = []
    gateAngles = []
    
    # perform object detection within each ROI
    for roi in rois:
        # crop frame to ROI and detect objects within ROI
        roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
        results = model(roi_frame)

        xyValue, xyName, xyConfi = grabInf(results)
        # Object Detection box
        i = 0
        for obj in xyValue:
            xMin, yMin, xMax, yMax = obj[0][0]+roi[0], obj[0][1]+roi[1], obj[1][0]+roi[0], obj[1][1]+roi[1]
            cv2.rectangle(frame, (xMin, yMin), (xMax, yMax ), (0, 0, 255), 2) 
            objectDetBoxs.append([xyName[i], xyConfi[i], xMin, yMin, xMax, yMax])
            i = i + 1
        # Draw ROI
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        #print(objectDetBoxs)
        
        try:
            highConf, angleDegree = getAngle(objectDetBoxs)
        except Exception as e:
            print(f'ERROR getAngle(): {e}')
            continue
        #highConf, angleDegree = getAngle(objectDetBoxs)
        gateAngles.append(angleDegree)

    try:
        conclusion, num_close, num_open = checkStatus(gateAngles, 45.0)
    except Exception as e:
        print(f'ERROR checkStatus(): {e}')
    #print(conclusion)

    # Put Text on the video
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Current Status: {conclusion}, NumOfClose: {num_close}, NumOfOpen: {num_open}', (10, 50), font, 1, (0,0,139), 2, cv2.LINE_AA)
    
    # Get video playing position
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    # print current time
    print(f'Current time: {round(current_time, 2) }s, Current status: {conclusion}')
    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
