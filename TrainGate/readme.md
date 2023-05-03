# ROI_DetectGate_v2

This repository contains a train gate detection algorithm that uses OpenAI's YOLOv5 to identify whether train gates are raised or lowered in a provided video. Additionally, the algorithm outputs a file that records when the train gate is closed or opened.

## Files

The repository consists of two main files:

1. `ROI_Select.py`: A utility file for selecting Regions of Interest (ROIs) in the video. ![ROI Example](ROI.PNG)
2. `ROI_DetectGate_v2.py`: The main file containing the implementation of the train gate detection algorithm. ![Detect Example](Detect.PNG)

## Algorithm

The algorithm works as follows:

1. Select multiple ROIs in the video using the `ROI_SelectArea` function from the `ROI_Select.py` file.
2. For each ROI, perform object detection using YOLOv5.
3. Calculate the angle of the detected object and determine if the gate is open or closed.
4. Check if the conclusion (open or closed) remains the same for a specified duration (default is 2 seconds).
5. Output the status (open or closed) along with the current time in the video to a text file.


## About Selecting ROI using `ROI_Select.py`

During the ROI selection process, you can interact with the video window using the following key commands:

- **Left Mouse Button**: Click and drag the left mouse button to draw a rectangle defining the ROI. Release the button to complete the selection.

- **q**: Press the 'q' key to quit the ROI selection process.

- **r**: Press the 'r' key to reset the ROI selections. This will clear all the drawn rectangles and allow you to start over with the selection process.

- **s**: Press the 's' key to save the selected ROIs and proceed with the video analysis. The algorithm will only focus on the areas within the selected ROIs to detect the train gate status.

Once you have selected the ROIs and pressed the 's' key, the program will analyze the video frames and output the detected gate status (open or closed) along with the current time in the video to a text file with the same name as the video file in the same directory.

## Fast-Forwarding and Rewinding the Video

During the video analysis process, you have the ability to fast-forward or rewind the video using the following key commands:

- **d**: Press the 'd' key to move the video forward by 10 seconds. This will allow you to quickly navigate to a specific point in the video for more precise analysis.

- **a**: Press the 'a' key to move the video backward by 10 seconds. This will enable you to review a previous section of the video or to fine-tune your ROI selections.

- **q**: Press the 'q' key to quit the video analysis process. The script will stop running, and any gate status detections up to this point will be saved to the output text file.


## Usage

1. Ensure that you have the necessary dependencies installed, including OpenCV, PyTorch, and YOLOv5.
2. Run `ROI_DetectGate_v2.py` to execute the train gate detection algorithm. The script will prompt you to select ROIs in the video.
3. After selecting the ROIs, the algorithm will analyze the video and output the status (open or closed) of the gates along with the current time in the video to a text file.

## Dependencies

- OpenCV
- PyTorch
- YOLOv5


