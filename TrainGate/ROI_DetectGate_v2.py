import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
import time
import os


from ROI_Select import ROI_SelectArea

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='trainresult/last_allGate_onlyorigin_0223.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='trainresult/last.pt', force_reload=True)
model.conf = 0.15
videofile = "test_im&Vde/6255092968981447d491ef3a_1671631820.mp4"
videofile = "test_im&Vde/Peek62ed249f02fcc15ffec0ba49_1672577344.mp4"
#videofile = "test_im&Vde/test_video_daytime.mp4"

# GreenStreet
#videofile = 'algorthimn_limit/video/GreenStreet/GreenStreet_daytime.mp4'

# NorthClintonStreet
#daytime
#videofile = 'algorthimn_limit/video/NorthClintonStreet/NorthClintonStreet_daytime.mp4'
#videofile = 'algorthimn_limit/video/NorthClintonStreet/NorthClintonStreet_nighttime.mp4'
#Vincennes:
#videofile = 'algorthimn_limit/video/Vincennes/Vincennes_daytime.mp4'
#videofile = 'algorthimn_limit/video/Vincennes/Vincennes_nighttime.mp4'


# get infor from "Result", we can get bounding box value, name, ... in this function
def grabInf(results):
    Inf = results.pandas().xyxy[0]
    boxXY, boxName, boxConfi = [], [], []
    for index, row in Inf.iterrows():
        xyMin, xyMax = (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax']))
        boxXY.append((xyMin, xyMax))
        boxName.append(row['name'])
        boxConfi.append(row['confidence'])
        # print(Inf)
    # print(boxName)
    return boxXY, boxName, boxConfi


# Using the highest confidence detection to calculate angle
def getAngle(objectDetBoxs):
    if objectDetBoxs == [] or objectDetBoxs == [[]]:
        return []
    # name, confidence,xtopletf, ytopleft, xdownright, ydownright
    i, length = 0, len(objectDetBoxs)
    highConf, angleDegree = 0, 0

    while i < length:
        objectDetBox = objectDetBoxs[i]
        objName, objConf, xMin, yMin, xMax, yMax = objectDetBox[0], objectDetBox[1], objectDetBox[2], objectDetBox[3], \
                                                   objectDetBox[4], objectDetBox[5]
        if objConf > highConf:
            highConf = objConf
            height = yMax - yMin
            distance = xMax - xMin
            angleDegree = math.degrees(math.atan(height / distance))
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

# Checks if the conclusion remains the same for ? seconds or not
def check_conclusion_stability(current_time, conclusion, prev_time_conclusion, stability_duration=2):
    if prev_time_conclusion is None:
        return None, (current_time, conclusion)

    time_difference = current_time - prev_time_conclusion[0]

    if time_difference >= stability_duration:
        if conclusion == prev_time_conclusion[1]:
            return conclusion, (current_time, conclusion)
        else:
            return None, (current_time, conclusion)

    return None, prev_time_conclusion
# default
prev_time_conclusion = None
def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ROI Multiple
rois = ROI_SelectArea(videofile)
cap = cv2.VideoCapture(videofile)


video_name = os.path.splitext(os.path.basename(videofile))[0]
# Create a new text file with the same name under the same folder
txt_file_path = os.path.join(os.path.dirname(videofile), f"{video_name}.txt")
with open(txt_file_path, 'a') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        objectDetBoxs = []
        gateAngles = []

        # If the frame is not read properly, break out of the loop
        if not ret:
            break

        # perform object detection within each ROI
        for roi in rois:
            # crop frame to ROI and detect objects within ROI
            try:
                roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
            except Exception as e:
                print(f'ERROR roi_frame() No ROI: {e}')
                break
            # roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
            results = model(roi_frame)

            xyValue, xyName, xyConfi = grabInf(results)
            # Object Detection box
            i = 0
            for obj in xyValue:
                xMin, yMin, xMax, yMax = obj[0][0] + roi[0], obj[0][1] + roi[1], obj[1][0] + roi[0], obj[1][1] + roi[1]
                cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (0, 0, 255), 2)
                objectDetBoxs.append([xyName[i], xyConfi[i], xMin, yMin, xMax, yMax])
                i = i + 1
            # Draw ROI
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
            # print(objectDetBoxs)

            try:
                highConf, angleDegree = getAngle(objectDetBoxs)
            except Exception as e:
                # print(f'ERROR getAngle(): {e}')
                continue
            # highConf, angleDegree = getAngle(objectDetBoxs)
            gateAngles.append(angleDegree)

        try:
            conclusion, num_close, num_open = checkStatus(gateAngles, 45.0)
        except Exception as e:
            print(f'ERROR checkStatus(): {e}')
        # print(conclusion)

        # Put Text on the video
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Current Status: {conclusion}, NumOfClose: {num_close}, NumOfOpen: {num_open}', (10, 50), font,
                    1, (0, 0, 139), 2, cv2.LINE_AA)

        # Get video playing position
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # print current time
        #print(f'Current time: {round(current_time, 2)}s, Current status: {conclusion}')
        cv2.imshow('Frame', frame)



        # Check if the conclusion is stable for 2 seconds
        stable_conclusion, prev_time_conclusion = check_conclusion_stability(current_time, conclusion, prev_time_conclusion, 1)

        if stable_conclusion is not None:
            print(f'Current time: {format_time(round(current_time, 3))}, conclusion: {stable_conclusion}')
            formatted_time = format_time(current_time)
            f.write(f'{stable_conclusion} {formatted_time}\n')



        # Wait for user input
        key = cv2.waitKey(10) & 0xFF
        # Move the video forward/backward by 10 seconds if the user presses the right/left arrow keys
        if key == ord('d'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 10 * cap.get(cv2.CAP_PROP_FPS))
        elif key == ord('a'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 10 * cap.get(cv2.CAP_PROP_FPS))
        # Break the loop if the user presses the 'q' key or no more Frame
        elif key == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    cap.release()
    cv2.destroyAllWindows()
    f.close()
