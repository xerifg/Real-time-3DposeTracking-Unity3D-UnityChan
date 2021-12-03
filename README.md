# Result

![result](https://github.com/xerifg/Real-time-3DposeTracking-Unity3D-UnityChan/blob/master/pictures/3D%E4%BB%BF%E7%9C%9F.gif)

# Environment

* Ubuntu20.04.01
* mediapipe  0.8.8
* opencv-python  4.1.2.30
* pyqt5  5.15.4
* python  3.8.8
* Unity3D-2020.3
* camera: D435 or Webcamera or any RGB camera

# Acknowledge

Firstly I use [Mediapipe](https://google.github.io/mediapipe/) to get human 2D pose, secondly trained a [Network](https://arxiv.org/abs/1705.03098) to get 3Ddata from 2Ddata (the results of last step), then sava the 3Dpose data (3d_data0.txt). Finally using Unity3D read the data and simulate the 3Dpose by Unity-Chan. **The FPS is about 25fps**

# Usage

1. run human pose data capture script

   ```shell
   python Getposedata/ActionRecorder_webcamera.py
   ```

   You will find a *3d_data0.txt* in Assets/datats. The Unity3D will read data from here.

   By the way, you can capture origin picturers using **D435** in this project, you just need to run **ActionRecorder_d435.py** instead of ActionRecorder_webcamera.py

2. open this project in Unity3D and  click run.

# Progress

### 1. Get 2D pose data

l use **mediapipe** to get 2Dpose data in my project, in fact mediapipe can get 3Dpose data directly. However i found its depth is not accurate very well. So i used an another network to estimate 3Dpose from 2Dpose. 

### 2. From 2Dpose to 3Dpose

The paper site: [(2017)A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/abs/1705.03098)

l have gotten the network parameters (*ckpt_best.pth.tar*) in *Getposedata/src*. so  you don't need to train network again.

If you want to train network by yourself, you can do it as follow:

1. run *src/DataConvert.py*, then you will get a file(*train_s1.pth.tar*). lt means a dataset from *S1*,(in data/ you can find S1, S5, S6, S7, S8, S9, S11). The data is from **Human3.6M**. If you want to increase number of train data, you can change it in *DataConver.py*, its code is very easy, believe me. My train data is **S1**, and my test data is **S9, S11**.
2. run *src/trainModel.py*, then you will get **ckpt_best.pth.tar** in *src/*, the ActionRecorder_webcamera.py will load it as **network parameters**

### 3. Simulate in Unity3D

load this project in Unity3D, and open *Assets/UnityChan/Scenes/Locomotion.unity*, finally click button run.Note : you must run ActionRecorder_webcamera.py before Unity3D, because the Unity3D needs 3Dpose datas from ActionRecorder_webcamera.py result.

# Unity3D

l control the unity-chan(3D model) by **FullbodyIK**. In short, you will create a Gameobject named FullBodyIK, there are 12 effectors in it, (Hips, Neck, LeftArm, LeftElbow, LeftWrist, RightArm, RightElbow, RightWrist, LeftKnee, LeftFoot, RightKnee, RightFoot), you will get 17(x,y,z) points data from ActionRecorder_webcamera.py result. The script of Unity3D will control 12 effectors(Gameobject) using 17 points location. Then 12 effectors will control animation's Skeleton, the animation(Unity-chan) will move according to  3Dpose datas(17 points  real-time location).

 You can learn Inverse Kinematics(**IK**) Unity3D from [here](https://www.youtube.com/watch?v=Kk0xN26ICLQ&t=526s)

# About knowledge

* The video about [ how to install Unity and use it?](https://www.bilibili.com/video/BV18x411X7ds?spm_id_from=333.999.0.0)

* You can learn how to use mediapipe from [here](https://www.youtube.com/watch?v=brwgBf6VB0I&t=434s)

* [The explanation of Human3.6M data structure](https://www.it610.com/article/1295032957748191232.htm)

* how to control unity-chan [Blog1](https://qiita.com/fuloru169/items/aa2960ea2601fd94b25f), [Blog2](https://qiita.com/kenkra/items/7b5634ff7f8c6bf0257a)

# Questions

This project only can simulate human 3Dpose now, it can't simulate hands pose and face emotion, in fact l want to add hand pose, but i find the FingerIK is nothing in FullBodyIK.cs in Unity3D, and l'm not good at Unity3D.

**If you know how to improve the Unity scripts, i will be very happy talking about it with you.**

