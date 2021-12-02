import numpy as np
import math
from sympy import *
import time

def distance_twopoints(a, b):
    distance = math.sqrt(math.pow((a[0] - b[0]), 2) + math.pow((a[1] - b[1]), 2))
    return distance


# def rotateMatrix(a, b, l_move, T_up):
#     cos = Symbol('cos')
#     sin = Symbol('sin')
#     T = np.array([[cos, -sin, l_move], [sin, cos, 0], [0, 0, 1]])
#     T2 = T_up @ T
#     # print(T_up)
#     y1 = b[0] - (T2[0][0] * a[0] + T2[0][1] * a[1] + T2[0][2] * 1)
#     y2 = b[1] - (T2[1][0] * a[0] + T2[1][1] * a[1] + T2[1][2] * 1)
#     sol = solve([y1, y2], [cos, sin])
#     T = np.array([[sol[cos], -sol[sin], l_move], [sol[sin], sol[cos], 0], [0, 0, 1]])
#     # print(T)
#     return T


# T_up = [a, b, c;
#         d, e, f;
#         0, 0, 1]
# T = [cos, -sin, l_move;
#     sin, cos, 0;
#     0, 0, 1]
# a = [x, y]
# b = [x^,y^]
def rotateMatrix(a, b, l_move, T_up):
    axby = T_up[0][0] * a[0] + T_up[0][1] * a[1]
    if axby == 0:
        axby = 1e-10
    ay_bx = T_up[0][0] * a[1] - T_up[0][1] * a[0]
    alc = T_up[0][0] * l_move + T_up[0][2]
    dxey = T_up[1][0] * a[0] + T_up[1][1] * a[1]
    ex_dy = T_up[1][1] * a[0] - T_up[1][0] * a[1]
    dlf = T_up[1][0] * l_move + T_up[1][2]
    sin = (b[1]*axby - dxey * b[0] + dxey * alc - dlf * axby)/(dxey * ay_bx + ex_dy * axby)
    cos = (b[0] + ay_bx * sin - alc) / (axby)
    T = np.array([[cos, -sin, l_move], [sin, cos, 0], [0, 0, 1]])
    return T

# x = [0.0, 132.9486, 100.218094, 74.69956, -132.94884, -211.95801, -288.76022, 31.811867, 60.47164, 82.537994,
# 98.816666, -84.927956, -363.0482, -604.4145, 209.25607, 481.22766, 720.7107]
# y = [0.0, 0.0, -439.15747, -892.54767, 0.0, -432.70737, -879.94006, 225.73943, 474.72375, 590.6828 ,
# 675.6102, 483.21246, 480.6143, 522.3414, 449.10007, 387.6592, 382.7377]
x = [0, 1, 1, 1,-1,-1,-1,0,0,0,0,-1,-2,-3,1,2,3]
y = [0, 0,-1, -2,0,-1,-2,1,2,3,4,2,2,2,2,2,2]
T = [] # all rotate matrix:# T01, T12, T23, T04, T45, T56, T07, T78, T89, T910 ,T811, T1112, T1213, T814, T1415, T1516
T_direct = [] # T01, T02, T03, T04, T05, T06, T07, T08, T09, T010, T011, T012, T013, T014, T015, T016
def change_skele_length(x, y):
    x_new, y_new = [0.0], [0.0]
    skeleton_length_old = [0.0]
    skeleton_length_new = [0, 132.95, 442.89, 454.21, 132.95, 442.89, 454.21, 233.38, 257.08, 121.13, 115,
                           151.03, 278.88, 251.73, 151.03, 278.88, 251.73]
    # get current skeleton length
    for i in range(len(x) - 1):
        if i == 10:
            n1 = 8
            n2 = 11
        elif i == 13:
            n1 = 8
            n2 = 14
        elif i == 3:
            n1 = 0
            n2 = 4
        elif i == 6:
            n1 = 0
            n2 = 7
        else:
            n1 = i
            n2 = i + 1
        a = [x[n1], y[n1]]
        b = [x[n2], y[n2]]
        distance = distance_twopoints(a, b)
        skeleton_length_old.append(distance)
    # print('skeleton_length_old:',skeleton_length_old)
    # print('skeleton_length_new: ',skeleton_length_new)
    # get current rotate matrix
    # for i in range(len(x)):
        # if i+1 == 1 or i+1 == 4 or i+1 == 7:
        #     a = np.array([skeleton_length_old[i+1], 0])
        #     b = np.array([x[i+1],y[i+1]])
        #     l_move = skeleton_length_old[0]
        #     T_last = np.eye(3)
        #     T.append(rotateMatrix(a, b, l_move, T_last))
        # else:
        #     a = np.array([skeleton_length_old[i + 1], 0])
        #     b = np.array([x[i + 1], y[i + 1]])
        #     l_move = skeleton_length_old[i]
        #     T_last_all = np.eye(3)
        #     T.append(rotateMatrix(a, b, l_move, T_last_all))
    #1
    a = np.array([skeleton_length_old[1], 0])
    b = np.array([x[1],y[1]])
    l_move = skeleton_length_old[0]
    T_up = np.eye(3)
    T01 = rotateMatrix(a, b, l_move, T_up)
    #update
    T01_new = T01.copy()
    T01_new[0][2] = skeleton_length_new[0]
    p11 = np.array([skeleton_length_new[1], 0, 1])
    p01 = T01_new @ p11
    x_new.append(float(p01[0]))
    y_new.append(float(p01[1]))
    #2
    a = np.array([skeleton_length_old[2], 0])
    b = np.array([x[2],y[2]])
    l_move = skeleton_length_old[1]
    T12 = rotateMatrix(a, b, l_move, T01)
    T02 = T01 @ T12
    #update
    T12_new = T12.copy()
    T12_new[0][2] = skeleton_length_new[1]
    T02_new = T01_new @ T12_new
    p22 = np.array([skeleton_length_new[2], 0, 1])
    p02 = T02_new @ p22
    x_new.append(float(p02[0]))
    y_new.append(float((p02[1])))

    #3
    a = np.array([skeleton_length_old[3], 0])
    b = np.array([x[3], y[3]])
    l_move = skeleton_length_old[2]
    T23 = rotateMatrix(a, b, l_move, T02)
    T03 = T02 @ T23
    # update
    T23_new = T23.copy()
    T23_new[0][2] = skeleton_length_new[2]
    T03_new = T02_new @ T23_new
    p33 = np.array([skeleton_length_new[3], 0, 1])
    p03 = T03_new @ p33
    x_new.append(float(p03[0]))
    y_new.append(float((p03[1])))

    #4
    a = np.array([skeleton_length_old[4], 0])
    b = np.array([x[4], y[4]])
    l_move = skeleton_length_old[0]
    T_up = np.eye(3)
    T04 = rotateMatrix(a, b, l_move, T_up)
    # update
    T04_new = T04.copy()
    T04_new[0][2] = skeleton_length_new[0]
    p44 = np.array([skeleton_length_new[4], 0, 1])
    p04 = T04_new @ p44
    x_new.append(float(p04[0]))
    y_new.append(float(p04[1]))

    #5
    a = np.array([skeleton_length_old[5], 0])
    b = np.array([x[5], y[5]])
    l_move = skeleton_length_old[4]
    T45 = rotateMatrix(a, b, l_move, T04)
    T05 = T04 @ T45
    # update
    T45_new = T45.copy()
    T45_new[0][2] = skeleton_length_new[4]
    T05_new = T04_new @ T45_new
    p55 = np.array([skeleton_length_new[5], 0, 1])
    p05 = T05_new @ p55
    x_new.append(float(p05[0]))
    y_new.append(float(p05[1]))

    #6
    a = np.array([skeleton_length_old[6], 0])
    b = np.array([x[6], y[6]])
    l_move = skeleton_length_old[5]
    T56 = rotateMatrix(a, b, l_move, T05)
    T06 = T05 @ T56
    # update
    T56_new = T56.copy()
    T56_new[0][2] = skeleton_length_new[5]
    T06_new = T05_new @ T56_new
    p66 = np.array([skeleton_length_new[6], 0, 1])
    p06 = T06_new @ p66
    x_new.append(float(p06[0]))
    y_new.append(float((p06[1])))

    #7
    a = np.array([skeleton_length_old[7], 0])
    b = np.array([x[7], y[7]])
    l_move = skeleton_length_old[0]
    T_up = np.eye(3)
    T07 = rotateMatrix(a, b, l_move, T_up)
    # update
    T07_new = T07.copy()
    T07_new[0][2] = skeleton_length_new[0]
    p77 = np.array([skeleton_length_new[7], 0, 1])
    p07 = T07_new @ p77
    x_new.append(float(p07[0]))
    y_new.append(float((p07[1])))

    #8
    a = np.array([skeleton_length_old[8], 0])
    b = np.array([x[8], y[8]])
    l_move = skeleton_length_old[7]
    T78 = rotateMatrix(a, b, l_move, T07)
    T08 = T07 @ T78
    # update
    T78_new = T78.copy()
    T78_new[0][2] = skeleton_length_new[7]
    T08_new = T07_new @ T78_new
    p88 = np.array([skeleton_length_new[8], 0, 1])
    p08 = T08_new @ p88
    x_new.append(float(p08[0]))
    y_new.append(float((p08[1])))

    #9
    a = np.array([skeleton_length_old[9], 0])
    b = np.array([x[9], y[9]])
    l_move = skeleton_length_old[8]
    T89 = rotateMatrix(a, b, l_move, T08)
    T09 = T08 @ T89
    # update
    T89_new = T89.copy()
    T89_new[0][2] = skeleton_length_new[8]
    T09_new = T08_new @ T89_new
    p99 = np.array([skeleton_length_new[9], 0, 1])
    p09 = T09_new @ p99
    x_new.append(float(p09[0]))
    y_new.append(float((p09[1])))

    # 10
    a = np.array([skeleton_length_old[10], 0])
    b = np.array([x[10], y[10]])
    l_move = skeleton_length_old[9]
    T910 = rotateMatrix(a, b, l_move, T09)
    T010 = T09 @ T910
    # update
    T910_new = T910.copy()
    T910_new[0][2] = skeleton_length_new[9]
    T010_new = T09_new @ T910_new
    p1010 = np.array([skeleton_length_new[10], 0, 1])
    p010 = T010_new @ p1010
    x_new.append(float(p010[0]))
    y_new.append(float(p010[1]))

    #11
    a = np.array([skeleton_length_old[11], 0])
    b = np.array([x[11], y[11]])
    l_move = skeleton_length_old[8]
    T811 = rotateMatrix(a, b, l_move, T08)
    T011 = T08 @ T811
    # update
    T811_new = T811.copy()
    T811_new[0][2] = skeleton_length_new[8]
    T011_new = T08_new @ T811_new
    p1111 = np.array([skeleton_length_new[11], 0, 1])
    p011 = T011_new @ p1111
    x_new.append(float(p011[0]))
    y_new.append(float(p011[1]))

    #12
    a = np.array([skeleton_length_old[12], 0])
    b = np.array([x[12], y[12]])
    l_move = skeleton_length_old[11]
    T1112 = rotateMatrix(a, b, l_move, T011)
    T012 = T011 @ T1112
    # update
    T1112_new = T1112.copy()
    T1112_new[0][2] = skeleton_length_new[11]
    T012_new = T011_new @ T1112_new
    p1212 = np.array([skeleton_length_new[12], 0, 1])
    p012 = T012_new @ p1212
    x_new.append(float(p012[0]))
    y_new.append(float(p012[1]))

    #13
    a = np.array([skeleton_length_old[13], 0])
    b = np.array([x[13], y[13]])
    l_move = skeleton_length_old[12]
    T1213 = rotateMatrix(a, b, l_move, T012)
    T013 = T012 @ T1213
    # update
    T1213_new = T1213.copy()
    T1213_new[0][2] = skeleton_length_new[12]
    T013_new = T012_new @ T1213_new
    p1313 = np.array([skeleton_length_new[13], 0, 1])
    p013 = T013_new @ p1313
    x_new.append(float(p013[0]))
    y_new.append(float(p013[1]))

    #14
    a = np.array([skeleton_length_old[14], 0])
    b = np.array([x[14], y[14]])
    l_move = skeleton_length_old[8]
    T814 = rotateMatrix(a, b, l_move, T08)
    T014 = T08 @ T814
    # update
    T814_new = T814.copy()
    T814_new[0][2] = skeleton_length_new[8]
    T014_new = T08_new @ T814_new
    p1414 = np.array([skeleton_length_new[14], 0, 1])
    p014 = T014_new @ p1414
    x_new.append(float(p014[0]))
    y_new.append(float(p014[1]))

    #15
    a = np.array([skeleton_length_old[15], 0])
    b = np.array([x[15], y[15]])
    l_move = skeleton_length_old[14]
    T1415 = rotateMatrix(a, b, l_move, T014)
    T015 = T014 @ T1415
    # update
    T1415_new = T1415.copy()
    T1415_new[0][2] = skeleton_length_new[14]
    T015_new = T014_new @ T1415_new
    p1515 = np.array([skeleton_length_new[15], 0, 1])
    p015 = T015_new @ p1515
    x_new.append(float(p015[0]))
    y_new.append(float(p015[1]))

    #16
    a = np.array([skeleton_length_old[16], 0])
    b = np.array([x[16], y[16]])
    l_move = skeleton_length_old[15]
    T1516 = rotateMatrix(a, b, l_move, T015)
    T016 = T015 @ T1516
    # update
    T1516_new = T1516.copy()
    T1516_new[0][2] = skeleton_length_new[15]
    T016_new = T015_new @ T1516_new
    p1616 = np.array([skeleton_length_new[16], 0, 1])
    p016 = T016_new @ p1616
    x_new.append(float(p016[0]))
    y_new.append(float(p016[1]))

    return x_new ,y_new


# start = time.time()
# xx ,yy = change_skele_length(x,y)
# end = time.time()
# print('time: ',end - start)
# print('x:',xx)
# print('y:',yy)

#
# print(type(xx[0]))