# import numpy as np
#
# a = np.array([2])
# b =a.copy()
# a[0]= 1
# print(a,b)
# st = ''
# for i in range(3):
#     st = st +'123' + '\r'
# print(st)
import numpy as np

# angles = np.arange(15).reshape(15,-1)
# expmapInd = np.split(np.arange(4,100)-1,32)
# # # print(expmapInd[0])
# # # for i in np.arange(32):
# # #     print(i)
# # parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
# #                        17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1
# # print(parent[6])
#
# print(angles.shape)
import torch
import os


# ckpt = torch.load(os.path.join('/home/xrf/ActionDetection/model', 'ckpt_best.pth.tar'))
# print(ckpt['state_dict'])
# a = [9,8,7,6,5,4,3,2,1]
# b = a[2::3]
# print(b)
# print(torch.cuda.is_available())
# a = torch.randn(3,3)
# b = np.array(a)
# b = np.double(b)
# print(b.dtype)
# a = np.array([[1,2,3]]).squeeze()
# a = np.array(['1','2'])
# b = np.array(['3','4'])
# c= np.vstack((a,b))
# w = np.array([])
# a = np.arange(15).reshape(3, 5)
# b = np.arange(15).reshape(3, 5)
# w = b
# c= np.vstack((a,w))
# from tqdm import tqdm
# for i in tqdm(range(1000)):
#      pass
#
# a =[1,2,3]
# print(len(a))
# for k in np.arange(0, 17 * 3, 3):
#     print(k)
# a = np.array([1,2,3])
# print(np.mean(a))
def saveData(datapose17):
    with open('/home/xrf/3D Pose Unity Project/Assets/datas/3d_data0.txt', 'w+') as f:
        dataString = "[["
        for i in range(0, 3):
            dataString = dataString + "["
            for j in range(0, 17):
                dataString = dataString + str(datapose17[j][i]) + ' '
                if (j + 1) % 5 == 0:
                    dataString = dataString + '\n'
            if i == 0 or i == 1:
                dataString = dataString + "]" + "\n"
            else:
                dataString = dataString + "]]]"
        f.write(dataString)


[inp,tar] = torch.load(os.path.join('/home/xrf/ActionDetection/src', 'test_s911.pth.tar'))
#
# # for i in range(tar.shape[0]):
saveData(tar[100].reshape(-1,3))
# print(tar['err'])

# a = np.array([[1,2,3],[4,5,6]])
# print(np.reshape(a,(1,-1)))

import math
from sympy import *


def distance_twopoints(a, b):
    distance = math.sqrt(math.pow((a[0] - b[0]), 2) + math.pow((a[1] - b[1]), 2))
    return distance


def rotateMatrix(a, b, l_move, T_up):
    cos = Symbol('cos')
    sin = Symbol('sin')
    T = np.array([[cos, -sin, l_move], [sin, cos, 0], [0, 0, 1]])
    T2 = T_up @ T
    y1 = b[0] - (T2[0][0] * a[0] + T2[0][1] * a[1] + T2[0][2] * 1)
    y2 = b[1] - (T2[1][0] * a[0] + T2[1][1] * a[1] + T2[1][2] * 1)
    sol = solve([y1, y2], [cos, sin])
    T = np.array([[sol[cos], -sol[sin], l_move], [sol[sin], sol[cos], 0], [0, 0, 1]])
    print(T)
    return T


# skel_length_old = [0]
# skel_length_new = [0, 2.82, 2.82, 2.82]
# coordinate_old = np.array([[0, 0], [1, 1], [2, 2], [4, 4]])
#
# for i in range(coordinate_old.shape[0] - 1):
#     distance = distance_twopoints(coordinate_old[i], coordinate_old[i + 1])
#     skel_length_old.append(distance)
#
# # first skeleton point
# a = np.array([skel_length_old[1], 0])
# b = coordinate_old[1]
# l_move = skel_length_old[0]
# T_up = np.eye(3)
#
# T01 = rotateMatrix(a, b, l_move, T_up)
#
# # second skeleton point
# a = np.array([skel_length_old[2], 0])
# b = coordinate_old[2]
# l_move = skel_length_old[1]
#
# T12 = rotateMatrix(a, b, l_move, T01)
# T02 = T01 @ T12
#
# # third skeleton point
# a = np.array([skel_length_old[3], 0])
# b = coordinate_old[3]
# l_move = skel_length_old[2]
#
# T23 = rotateMatrix(a, b, l_move, T02)
# T03 = T02 @ T23
#
# # update skeleton length
# T01[0][2] = skel_length_new[0]
# T01_new = T01
# T12[0][2] = skel_length_new[1]
# T02_new = T01_new @ T12
# T23[0][2] = skel_length_new[2]
# T03_new = T02_new @ T23
#
# p11 = np.array([skel_length_new[1], 0, 1])
# p01 = T01_new @ p11
#
# p22 = np.array([skel_length_new[2], 0, 1])
# p02 = T02_new @ p22
#
# p33 = np.array([skel_length_new[3], 0, 1])
# p03 = T03_new @ p33