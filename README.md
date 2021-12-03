# Environment

* mediapipe  0.8.8
* opencv-python  4.1.2.30
* pyqt5  5.15.4
* python  3.8.8
* Unity3D-2020.3
* camera: D435 or Webcamera or any RGB camera

# Usage

1. run human pose data capture script

   ```shell
   python Getposedata/ActionRecorder_webcamera.py
   ```

   You will find a *3d_data0.txt* in Assets/datats. The Unity3D will read data from here.

2. open your this project in Unity3D and  click run