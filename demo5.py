# 这一版本的程序，可以较好的显示出帧率的一些信息

import os
from glob import glob
import numpy as np
import cv2
import random
import copy
import time
from mecfTracker1 import mecfTracker1
import os
from glob import glob
import random
import copy
from ATL_path import scandir, mkdir_or_exist, find_data_list
from tqdm import tqdm, trange
import psutil
from PIL import Image


def gettxt(datapath):
    name = os.find_data_list(datapath, suffix='jpg')
    mat = []
    for i in range(len(name)):
        print(name[i])
        txtpath = glob(os.path.join(name[i],'groundtruth_rect*'))
        gt = np.loadtxt(txtpath[0], dtype=int, delimiter=',')
        print(txtpath,':  ',list(gt[0]))
        mat.append(list(gt[0]))
    return mat,name

def to_numpy(im):
    im.load()
    # unpack data
    e = Image._getencoder(im.mode, 'raw', im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast('B', (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


def get_image(image_path):
    img_lists = find_data_list(image_path, suffix='jpg')
    mat = []
    for idx in trange(len(img_lists),desc='正在将图像全部读入内存ing...',colour='GREEN'):
        # frame_copy = cv2.imread(img_lists[idx])  #这一步骤就是把图全读进来了，别这样把
        frame_copy = Image.open(img_lists[idx])
        frame_copy = to_numpy(frame_copy)
        mat.append(frame_copy)
    
    return mat

def Images_to_Video(Images_list, VideoSavePath, frame_rate=10):

    """ 把一串图像序列，保存为视频输出

    Args:
        Images_list: 经过跟踪算法后存储的,带有 bbox 的图像列表,每一个列表是一张图
        VideoSavePath:输出视频的路径
        frame_rate:输出视频的帧率
    
    Return:
        None,直接把视频保存到了 VideoSavePath 中

    """
    print(f'--------------------------------------------------------------')
    img_shape = Images_list[0].shape
    width, height = img_shape[0], img_shape[1]

    # 创建视频编码器
    video_name = os.path.join(VideoSavePath,'output_video.mp4')
    # video_name = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要更改编码器（如：XVID、MJPG等）
    video_out = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    # 将每个图像逐帧写入视频
    for frame in trange(len(Images_list),desc=' --正在将图像序列写入视频中...',colour='GREEN'):
        video_out.write(Images_list[frame])

    # 释放资源
    video_out.release()
    print(' --视频已保存完成')

def trackmethod(img_mat, original_position, savepath):
    
    # save txt file
    txtpath = os.path.join(savepath,'txt')
    mkdir_or_exist(txtpath)

    # save track image file 
    outimgpath = os.path.join(savepath, 'output')
    mkdir_or_exist(outimgpath)

    mat3 = []
    high, width, channel = img_mat[0].shape     # img_mat 列表,每一个元素是一张图

    track_timestamp_start = time.time()         # 开始跟踪的时间
    for ii in range(len(original_position)):    # 开始循环跟踪16个位置的船
        # print(name[ii])
        a = time.time()
        tracker = mecfTracker1()
        bbox = list(original_position[ii])
        # print(bbox)
        mat2 = []
        for mm in range(len(img_mat)):
            if mm == 0:
                tracker.init(img_mat[mm], bbox)
                mat2.append(bbox)
            else:
                x, y, w, h = bbox
                if x < 0 or y < 0 or x + w > width or y + h > high:
                    break
                # print(bbox)
                ok, bbox = tracker.update(img_mat[mm])
                bbox = list(map(int, map(np.round, bbox)))
                mat2.append(bbox)
        b = time.time() - a
        # print('FPS:',len(img_mat)/b)
        txt = os.path.join(txtpath, str(ii+1).zfill(4)+'.txt')
        # txt2 = os.path.join(txtpath2, name[ii] + '.txt')
        np.savetxt(txt,np.array(mat2),fmt='%d',delimiter=',')
        # np.savetxt(txt2, np.array(mat2), fmt='%d', delimiter=',')
        mat3.append(mat2)
    track_timestamp_end = time.time()
    
    print(f'--------------------------------------------------------------')
    print(f' --跟踪处理完成')
    print(f' --跟踪总帧数：{len(img_mat)}')
    print(f' --跟踪总耗时：{round(track_timestamp_end-track_timestamp_start, 3)} 秒')
    print(f' --平均每帧跟踪耗时：{round((track_timestamp_end-track_timestamp_start)/len(img_mat)*1000, 3)} ms')
    print(f' --平均跟踪帧率：{round(1/((track_timestamp_end-track_timestamp_start)/len(img_mat)), 3)} FPS')
    print(f'--------------------------------------------------------------')


    color = []
    outputvideo = copy.copy(img_mat)
    for qq in range(len(mat3)):
        p = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        color.append(p)

    
    track_img = []
    for jj in range(len(outputvideo)):
        for kk in range(len(mat3)):
            lent = len(mat3[kk])
            if lent <= jj:
                continue
            list2 = mat3[kk][jj]
            cv2.rectangle(outputvideo[jj], (list2[0], list2[1]), (list2[0] + list2[2], list2[1] + list2[3]), color[kk], 3)
        track_img .append(outputvideo[jj])
    return track_img
        # cv2.imwrite(os.path.join(outimgpath, str(jj+1).zfill(6)) + '.jpg', outputvideo[jj])


if __name__ == '__main__':

    img_path = './JL1_1/'                    # 存储图像的地方
    result_path = './Track_result_video/'    # 存储最后视频结果的地方
    mkdir_or_exist(result_path)              # 创建存储结果的文件夹

    # 船只的初始位置，这里是由目标检测算法来提供的
    original_position = [                  
            [4013, 4395, 13, 13], [4317, 3757, 10, 8], 
            [1117, 3595, 38, 13], [1239, 3010, 14, 10], 
            [2285, 3297, 21, 12], [2634, 3023, 17, 19], 
            [3114, 1904, 25, 31], [3109, 2107, 25, 32], 
            [3476, 1555, 19, 12], [4000, 556,  38, 64], 
            [4785, 671, 11, 17],  [3913, 3827, 20, 38], 
            [3541,4046,16,12]]

    # Read Img From Disk
    read_img_time_start = time.time()
    img_mat = get_image(img_path)       # Read Img From Disk
    read_img_time_end = time.time()
    img_read_time = read_img_time_end-read_img_time_start
    print(f' --读取图像至内存共耗时：{round(img_read_time, 5)} s')
    


    # Track 
    track_img_mat = trackmethod(img_mat, original_position, result_path)

    print(f' --读取图像至内存共耗时：{round(img_read_time, 5)} s')
    
    # Get Memory Info
    process = psutil.Process()
    mem_info = process.memory_info()
    print (' --当前进程的内存使用：',psutil.Process(os.getpid()).memory_info().rss)
    print (' --当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    # Generate Video File
    Images_to_Video(track_img_mat, result_path, 10)


