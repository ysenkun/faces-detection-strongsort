# Track Any Object Detection Model with StrongSORT

This repository can track any object detection model.  
In track.py, opencv's face detection is used as a sample for object detection.

### :raising_hand: Reference:
1. https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet

## Seting Up Environment

Download the model for face detection in opencv.
```bash
$ wget -nc https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml -O ./haarcascade_frontalface_default.xml
```

```bash
$ pip3 install -r requirements.txt
```
### 

## Run
Enter the video-path and run it.
```bash
$ python3 track.py --source vid.mp4 # video path
```


