import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import data_utils
import viz
from tqdm import tqdm


def fkl(angles, parent, offset, expmapInd):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.
    Based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
    Args
      angles: 99-long vector with 3d position and 3d joint angles in expmap format
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    Returns
      xyz: 32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        r = angles[expmapInd[i]]

        thisRotation = data_utils.expmap2rotmat(r)  # get rotation matrix

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3))  # （0，0，0）
        else:
            xyzStruct[i]['xyz'] = (offset[i, :]).dot(xyzStruct[parent[i]]['rotation']) + xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]

    return np.reshape(xyz, [-1])


def revert_coordinate_space(channels, R0, T0):
    """
    Bring a series of poses to a canonical form so they are facing the camera when they start.
    Adapted from
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m
    Args
      channels: n-by-99 matrix of poses
      R0: 3x3 rotation for the first frame
      T0: 1x3 position for the first frame
    Returns
      channels_rec: The passed poses, but the first has T0 and R0, and the
                    rest of the sequence is modified accordingly.
    """
    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for ii in range(n):
        R_diff = data_utils.expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
        T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii, :3], [3, 1]))).reshape(-1)
        channels_rec[ii, :3] = T
        T_prev = T
        R_prev = R

    return channels_rec


def _some_variables():
    """
    We define some variables that are useful to run the kinematic tree
    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 251.733451, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, expmapInd


def main():
    inp = np.array([])
    out = np.array([])
    isdraw = False # if draw skeleton
    path_datas = os.path.abspath(os.path.dirname(os.getcwd()))
    data_path = f'{path_datas}/data/'
    # # read data
    dir = ['S1']
    fname = ['directions_1', 'discussion_1', 'eating_1', 'greeting_1', 'phoning_1', 'posing_1', 'purchases_1',
             'sitting_1', 'sittingdown_1', 'smoking_1', 'takingphoto_1', 'waiting_1', 'walking_1', 'walkingdog_1',
             'walkingtogether_1',
             'directions_2', 'discussion_2', 'eating_2', 'greeting_2', 'phoning_2', 'posing_2', 'purchases_2',
             'sitting_2', 'sittingdown_2', 'smoking_2', 'takingphoto_2', 'waiting_2', 'walking_2', 'walkingdog_2',
             'walkingtogether_2'
             ]
    # fname = ['sitting_1', 'sitting_2']
    x_data = [0, 3, 6, 9, 18, 21, 24, 36, 39, 42, 45, 51, 54, 57, 75, 78, 81]
    y_data = [1, 4, 7, 10, 19, 22, 25, 37, 40, 43, 46, 52, 55, 58, 76, 79, 82]
    z_data = [2, 5, 8, 11, 20, 23, 26, 38, 41, 44, 47, 53, 56, 59, 77, 80, 83]
    # Load all the data
    parent, offset, expmapInd = _some_variables()

    # numpy implementation
    for di in dir:
        for fn in fname:
            print("processing: ", di, "/", fn, "(", fname.index(fn) + 1, "/", len(fname), ")")
            input_2d, output_3d = [], []
            with open(data_path + di + '/' + fn + '.txt', 'r+', encoding='utf-8') as f:
                s = [i[:-1].split(',') for i in f.readlines()]
            expmap_gt = np.array(s)  # from list to numpy
            expmap_gt = expmap_gt.astype(float)
            nframes_gt = expmap_gt.shape[0]  # the line number of dataset

            # Put them together and revert the coordinate space
            expmap_gt = revert_coordinate_space(expmap_gt, np.eye(3), np.zeros(3))

            # Compute 3d points for each frame
            xyz_gt = np.zeros((nframes_gt, 96))
            for i in tqdm(range(nframes_gt)):
                xyz_gt[i, :] = fkl(expmap_gt[i, :], parent, offset, expmapInd)

            # === Plot and animate ===
            if isdraw:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ob = viz.Ax3DPose(ax)

                # Plot the conditioning ground truth
                for i in tqdm(range(nframes_gt)):
                    ob.update(xyz_gt[i, :])
                    plt.show(block=False)
                    fig.canvas.draw()
                    plt.pause(0.01)

            # save data to local
            for i in range(nframes_gt):
                # save data
                for j in range(17):
                    input_2d.append(float(xyz_gt[i][x_data[j]]))
                    input_2d.append(float(xyz_gt[i][z_data[j]]))
                    output_3d.append(float(-xyz_gt[i][x_data[j]]))  # x: weight
                    output_3d.append(float(-xyz_gt[i][y_data[j]]))  # y: depth
                    output_3d.append(float(xyz_gt[i][z_data[j]]))  # z: height

            input_2d = np.array(input_2d)
            input_2d = input_2d.reshape(-1, 17 * 2)
            input_2d = np.float32(input_2d)
            output_3d = np.array(output_3d)
            output_3d = output_3d.reshape(-1, 17 * 3)
            output_3d = np.float32(output_3d)

            if di == 'S1' and fn == 'directions_1':
                inp = input_2d
                out = output_3d
            else:
                inp = np.vstack((inp, input_2d))
                out = np.vstack((out, output_3d))

        torch.save((inp, out), 'train_s1.pth.tar')  # save dataset
    print("The shape of input: ", inp.shape)
    print("The shape of output: ", out.shape)


if __name__ == '__main__':
    main()
    # print()
# train_2d = torch.load(os.path.join(data_path, 'train_2d.pth.tar'))
# train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))
# stat_3d = torch.load(os.path.join(data_path, 'stat_3d.pth.tar'))
# # count = 0
# train_inp, train_out = [], []
# for k2d in train_2d.keys():
#     (sub, act, fname) = k2d
#     k3d = k2d
#     k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
#     assert train_3d[k3d].shape[0] == train_2d[k2d].shape[0], '(training) 3d & 2d shape not matched'
#     num_f, _ = train_2d[k2d].shape
#     for i in range(num_f):
#         train_inp.append(train_2d[k2d][i])
#         train_out.append(train_3d[k3d][i])
# #
# x_input, y_input, x_output, y_output, z_output = [], [], [], [], []
# for i in range(0, 32, 2):
#     x_input.append(train_inp[0][i])
#     y_input.append(train_inp[0][i+1])
# #
# for i in range(0, 48, 3):
#     x_output.append(train_out[0][i])
#     y_output.append(train_out[0][i+1])
#     z_output.append(train_out[0][i+2])

# def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
#     T = 1  # Batch size
#     D = data_mean.shape[0]  # 96
#
#     orig_data = np.zeros((T, D), dtype=np.float32)
#
#     orig_data[:, dimensions_to_use] = normalized_data
#
#     # Multiply times stdev and add the mean
#     stdMat = data_std.reshape((1, D))
#     stdMat = np.repeat(stdMat, T, axis=0)
#     meanMat = data_mean.reshape((1, D))
#     meanMat = np.repeat(meanMat, T, axis=0)
#     orig_data = np.multiply(orig_data, stdMat) + meanMat
#     return orig_data
#
# origin_3d = unNormalizeData(train_inp[0],stat_3d['mean'],stat_3d['std'], stat_3d['dim_use'])
# print(origin_3d.reshape(-1,3))

# # === Plot and animate ===
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ob = viz.Ax3DPose(ax)
# ob.update(origin_3d)
# plt.show(block=False)
# fig.canvas.draw()
# plt.pause(100)

# # plt.plot(x_input,y_input)
# plt.scatter(y_output,z_output)
# # plt.scatter(x_output,y_output)
# # fig = plt.figure()
# # ax = plt.gca(fc='whitesmoke',projection='3d')
# # ax.scatter(x_output,y_output, z_output)
# plt.show()

# fig = plt.figure()
# ax = plt.gca(projection='3d')
# # ob = viz.Ax3DPose(ax)
# fig.canvas.draw()

# print(stat_3d['mean'].reshape(-1,3))
# print(train_inp[0])
# print(train_out[0])
