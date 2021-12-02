from __future__ import print_function, absolute_import, division
import sys
# import math
import matplotlib.pyplot as plt
from src import viz
# from mpl_toolkits.mplot3d import Axes3D
from PyQt5 import QtCore, QtWidgets, uic, QtGui
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2
import os
import pickle
import time
import PoseTrackingModel as ptk
# import io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")

from CTransform import *

# Loading the UI window
qtCreatorFile = "InteractionUI.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

fig = plt.figure()

class Ax3DPose17(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,5,6,1,2,3,1,8,9,10, 9,12,13,9,15,16])-1
    self.J   = np.array([5,6,7,2,3,4,8,9,10,11,12,13,14,15,16,17])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    self.ax = ax

    vals = np.zeros((17, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]])
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]])
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]])
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.
    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 51, "channels should have 51 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (17, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 750
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('auto')


def pickle2(filename, data, compress=False):
    fo = open(filename, "wb")
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)  # 序列化对象，并将结果数据流写入到文件对象中
    fo.close()


def unpickle2(filename):
    fo = open(filename, 'rb')
    dict = pickle.load(fo)  # 反序列化对象,将文件中的数据解析为一个Python对象
    fo.close()
    return dict


def saverawimage(img):
    path = '/home/xrf/3D Pose Unity Project/Assets/webimage.jpg'
    cv2.imwrite(path,img)

def saveData(datapose17, num):
    with open(f'/home/xrf/RealtimeActionCapture/Assets/datas/3d_data{num}.txt', 'w+') as f:
        dataString = "[["
        for i in range(0, 3):
            dataString = dataString + "["
            for j in range(0, 17):
                dataString = dataString + str(datapose17[i][j]) + ' '
                if (j + 1) % 5 == 0:
                    dataString = dataString + '\n'
            if i == 0 or i == 1:
                dataString = dataString + "]" + "\n"
            else:
                dataString = dataString + "]]]"
        f.write(dataString)


# from 2d pose to 3d pose by network
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = 17 * 2
        # 3d joints
        self.output_size = 17 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(
            self.linear_stages)  ## 组装模型的容器，容器内的模型只是被存储在ModelList里并没有像nn.Sequential那样严格的模型与模型之间严格的上一层的输出等于下一层的输入

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        # post processing
        y = self.w2(y)

        return y


# from 2d to 3d
class From2dto3d():
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = LinearModel().to(self.device)  # 初始化模型
        self.model.apply(weight_init)  # 初始化模型参数
        self.checkpoint = torch.load(os.path.join('/home/xrf/ActionDetection/model', 'ckpt_best.pth.tar'))
        self.model.load_state_dict(self.checkpoint['state_dict'])

    def run(self, inputs):
        """

        :param inputs: 1*17*2
        :return: 1*17*3
        """
        self.model.eval()  # eval model
        # inputs = np.array(inputs)
        inputs = np.array(inputs)
        inputs = inputs.reshape(1, -1)
        # inputs = np.double(inputs)
        inputs = torch.tensor(inputs)
        inputs = inputs.to(torch.float32)
        inputs = Variable(inputs.cuda())  # convert to Variable type
        self.outputs = self.model(inputs)
        return self.outputs


class ActionRec(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        Set initial parameters here.
        Note that the demo window size is 1920*1080, you can edit this via Qtcreator.
        In this demo, we take 20 frames of profiles to generate a GEI. You can edit this number by your self.
        '''
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.showFullScreen()
        self.setupUi(self)
        # get camera images
        # self.capture = rsd.DepthCamera()  # through D435 camera
        self.capture = cv2.VideoCapture(0)  # through computer cam
        self.currentFrame = np.array([])
        self.originFrame = np.array([])
        self.thresh = np.array([])
        self.skeletonFrame = np.zeros((480, 640), dtype=np.uint8)
        self.poselist = []
        self.start_state = False
        self.save_state = False
        self.cTime = 0
        self.pTime = 0
        self.numframes = 0  ## the total number of valid frames
        self.detector = ptk.PoseDetector()
        self.landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]


        # Set two window for raw video and segmentation.
        self.video_lable = QtWidgets.QLabel(self.centralwidget)
        self.seg_label = QtWidgets.QLabel(self.centralwidget)
        self.skeleton_lable = QtWidgets.QLabel(self.centralwidget)
        self._timer = QtCore.QTimer(self)  # open Qt timer
        self.video_lable.setGeometry(0, 200, 640, 480)
        self.skeleton_lable.setGeometry(640, 200, 640, 480)
        self.seg_label.setGeometry(1280, 200, 640, 480)
        self._timer.timeout.connect(self.play)  # response function of Qt timer

        # Waiting for you to push the button.
        # The slot functions from Qt
        self.start.clicked.connect(self.start_record_slot)
        self.reset.clicked.connect(self.reset_slot)
        self.save.clicked.connect(self.save_data_slot)
        self._timer.start(27)  # the end time of Qt timer, it means get a frame to synthesis GEI every about 27 ms
        self.update()
        # instance 2d to 3d model
        self._from2dto3d = From2dto3d()

    def start_record_slot(self):
        '''
        To record the action data.
        '''
        self.start_state = True
        self.save_state = False
        self.state_print.setText('Recording ...')
        self.state_print.setAlignment(QtCore.Qt.AlignCenter)

    def reset_slot(self):
        self.save_state = False
        self.start_state = False
        self.state_print.clear()

    def save_data_slot(self):
        '''
        Save data to local.
        '''
        self.save_state = True
        self.start_state = False
        self.state_print.setText('Successfully Saved!')
        self.state_print.setAlignment(QtCore.Qt.AlignCenter)

    def cal_angle(self):
        pass

    def draw_skeleton(self):
        """
        draw the skeleton
        """
        self.skeletonFrame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.mpdraw.draw_landmarks(self.skeletonFrame, self.detector.results.pose_landmarks,
                                            self.detector.mppose.POSE_CONNECTIONS)

    def draw_boundingbox(self):
        '''
        draw the bounding box of human.
        '''
        # get mask
        annotated_image = np.zeros((480, 640), dtype=np.uint8)
        mask = self.detector.results.segmentation_mask  # dtype:float32
        condition = mask > 0.5  # element type is bool  threshold:0.5
        annotated_image[:] = 255
        bg_image = np.zeros(mask.shape, dtype=np.uint8)
        self.thresh = np.where(condition, annotated_image, bg_image)
        # Find the max box.
        sil_y_list, sil_x_list = (self.thresh > 100).nonzero()  # return the position of pix value>100
        x_topleft, y_topleft = sil_x_list.min(), sil_y_list.min()
        x_botright, y_botright = sil_x_list.max(), sil_y_list.max()
        w = x_botright - x_topleft
        h = y_botright - y_topleft
        max_rec = w * h
        if max_rec > 0:
            cv2.rectangle(self.originFrame, (x_topleft, y_topleft), (x_botright, y_botright), (0, 255, 0), 2)

    def transtoPose17(self, pose3D):
        global xx_new, yy_new
        datapose17 = [[], [], []]
        center = (np.array(pose3D[23]) + np.array(pose3D[24])) * 0.5
        chest = (np.array(pose3D[11]) + np.array(pose3D[12])) * 0.5
        spine = (center + chest) * 0.5
        mouth = (np.array(pose3D[9]) + np.array(pose3D[10])) * 0.5
        neck = (chest + mouth) * 0.5
        head = pose3D[0]
        for i in range(3):
            if i == 0:
                datapose17[0] = [0, pose3D[24][i], pose3D[26][i], pose3D[28][i], pose3D[23][i], pose3D[25][i],
                                 pose3D[27][i], spine[i], chest[i], neck[i],
                                 head[i], pose3D[11][i], pose3D[13][i], pose3D[15][i], pose3D[12][i], pose3D[14][i],
                                 pose3D[16][i]]
                for n, v in enumerate(datapose17[0]):
                    datapose17[0][n] = (center[i] - v)

            if i == 1:
                datapose17[2] = [0, pose3D[24][i], pose3D[26][i], pose3D[28][i], pose3D[23][i], pose3D[25][i],
                                 pose3D[27][i], spine[i], chest[i], neck[i],
                                 head[i], pose3D[11][i], pose3D[13][i], pose3D[15][i], pose3D[12][i], pose3D[14][i],
                                 pose3D[16][i]]

                for n, v in enumerate(datapose17[2]): datapose17[2][n] = (center[i] - v)

                # transform coordination
                datapose17[0] , datapose17[2] = change_skele_length(datapose17[0], datapose17[2])

            if i == 2:
                twoDposedata = []
                for i in range(17):
                    twoDposedata.append(datapose17[0][i])
                    twoDposedata.append(datapose17[2][i])
                threeDposedata = self._from2dto3d.run(twoDposedata)
                outputs_plot = threeDposedata
                threeDposedata = threeDposedata.data.cpu().numpy().squeeze().tolist()


                outputs_plot = outputs_plot.data.cpu().numpy().squeeze().copy()
                # print(outputs.shape)
                # # draw skelo
                #
                # ax = fig.add_subplot(projection='3d')
                # ob = Ax3DPose17(ax)
                #
                # # Plot the conditioning ground truth
                # ob.update(outputs_plot)
                # plt.show(block=False)
                # fig.canvas.draw()
                # plt.pause(0.001)
                # # plt.close()


                datapose17[1] = threeDposedata[1::3]
                # for n, v in enumerate(datapose17[0]): datapose17[0][n] = v * 1000
                for n, v in enumerate(datapose17[1]):datapose17[1][n] = v * -1
                # for n, v in enumerate(datapose17[2]): datapose17[2][n] = v * 1000

        return datapose17

    def play(self):
        '''
        Main program.
        '''
        ret, self.originFrame = self.capture.read()  # Read video from a camera.
        if (ret == True):
            """####################  preprocess  ########################"""
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # bgr to gray
            # gray = cv2.GaussianBlur(gray, (3, 3), 0)
            img_pose = self.detector.findPose(self.originFrame)  # get pose from frame

            self.poselist, self._pose2D, self._pose3D, self._DataPose3D = self.detector.findPosition(
                img_pose)  # get all landmarks position
            # feature_map = np.zeros((480, 640), np.single)
            # assert len(self.poselist) == 33, 'Unexpected number of landmarks: {}'.format(
            #     len(self.poselist))  # check number of landmarks that are detected
            if len(self.poselist) == 33:
                # print(np.array(transtoPose17(self._pose3D)).shape)
                DataPose17array = np.array(self.transtoPose17(self._pose3D))

                saveData(DataPose17array, self.numframes)  ## save 3D pose data

                # self.numframes  = self.numframes + 1
                # np.savetxt("output.txt",DataPose17array)
                if not self.start_state and not self.save_state:
                    self.state_print.setText('Working normally')
                    self.state_print.setAlignment(QtCore.Qt.AlignCenter)
                # draw skeleton
                self.draw_skeleton()
                # draw box
                self.draw_boundingbox()
                # calculate joints angles when click "start" button
                if self.start_state:
                    self.cal_angle()

                # Show results.
                self.currentFrame = cv2.cvtColor(self.originFrame, cv2.COLOR_BGR2RGB)
                self.currentSeg = Image.fromarray(self.thresh).convert('RGB')
                self.currentSeg = ImageQt(self.currentSeg)
                self.cur_skeletonFrame = Image.fromarray(self.skeletonFrame).convert('RGB')
                self.cur_skeletonFrame = ImageQt(self.cur_skeletonFrame)

                height, width = self.currentFrame.shape[:2]
                # show fps
                self.cTime = time.time()
                fps = 1 / (self.cTime - self.pTime)
                self.pTime = self.cTime
                cv2.putText(self.currentFrame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)

                img = QtGui.QImage(self.currentFrame,
                                   width,
                                   height,
                                   QtGui.QImage.Format_RGB888)

                img = QtGui.QPixmap.fromImage(img)

                self.video_lable.setPixmap(img)
                seg = QtGui.QImage(self.currentSeg)
                seg = QtGui.QPixmap(seg)
                self.seg_label.setPixmap(seg)

                skeleton = QtGui.QImage(self.cur_skeletonFrame)
                skeleton = QtGui.QPixmap(skeleton)
                self.skeleton_lable.setPixmap(skeleton)
            else:
                DataPose17array = np.zeros((3, 17))
                saveData(DataPose17array, self.numframes)
                self.state_print.setText('Failed to get landmarks!')
                self.state_print.setAlignment(QtCore.Qt.AlignCenter)

    def keyPressEvent(self, event):  # 重新实现了keyPressEvent()事件处理器。
        # 按住键盘事件
        # 这个事件是PyQt自带的自动运行的，当我修改后，其内容也会自动调用
        if event.key() == QtCore.Qt.Key_Escape:  # 当我们按住键盘是esc按键时
            self.close()  # 关闭程序


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ActionRec()
    window.show()
    sys.exit(app.exec_())
