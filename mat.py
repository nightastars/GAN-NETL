'''将.npy文件批量转换成.mat文件'''
'''该程序将处理成.npy文件的没有子文件夹的文件夹中批量转换成.mat——idwt'''

import os
import argparse
import numpy as np
import scipy.io as io  # 通过scipy.io转换：
from scipy import io
# import matplotlib.pyplot as plt
def save_dataset(args):
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        print('Create path : {}'.format(args.data_path))
    if not os.path.exists(args.data_path1):
        os.makedirs(args.data_path1)
        print('Create path : {}'.format(args.data_path1))
    a = []

    filenames = os.listdir(args.data_path)
    print('filenames1=', filenames)
    filenames.sort(key=lambda x: int(x[:-11]))
    print('filenames2=', filenames)  # 100_result.npy
    for i in filenames:
        name1 = os.path.join(args.data_path, i)
        a.append(name1)
        # print('a=',a)
    for i1 in range(len(a)):

        b = np.load(a[i1])
        f_name = '{}{}.mat'.format('c', i1)
        # io.savemat(os.path.join(args.data_path1, f_name), {('{}{}'.format('y', i1)): b})
        io.savemat(os.path.join(args.data_path1, f_name), {('{}'.format('c')): b})
        # io.savemat(os.path.join(args.data_path1, f_name), pp)

    # print(b.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='F:/图像合成/核/decompose_content0/')
    parser.add_argument('--data_path1', type=str, default='F:/图像合成/核/decompose_content0_MAT/')
    args = parser.parse_args()
    save_dataset(args)
