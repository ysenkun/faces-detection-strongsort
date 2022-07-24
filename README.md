# Track Faces Detection with StrongSORT

This repository can track faces.  
track.py uses opencv's face detection.

### :raising_hand: Reference:
1. https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet

## Seting Up Environment

```bash
$ git clone https://github.com/ysenkun/faces-detection-strongsort.git
```

Download the model for face detection in opencv.
```bash
$ wget -nc https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml -O ./haarcascade_frontalface_default.xml
```

```bash
$ pip3 install -r requirements.txt
```
### 

## Run
Enter the video path and run it.
```bash
$ python3 track.py --source vid.mp4 # video path
```
![face](https://user-images.githubusercontent.com/82140392/180652961-dc979cc1-d38d-427f-baba-3ad5b73d2a79.gif)

