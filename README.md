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
   python 3DPoseCapture/ActionCapture/ActionRecorder.py
   ```

   **Note:**You must change the funtion **saveData(datapose17, num)**  path to yours in ActionRecorder.py. The Unity3D will read data from here.

   ```python
   def saveData(datapose17, num):
       with open(f'/home/xrf/3D Pose Unity Project/Assets/data_Doit/{num}.txt', 'w+') as f:
           dataString = "[["
           ......
   ```

   

2. open your Unity3D and  click run