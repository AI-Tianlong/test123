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
 
 
"""
这一个版本，可以生成视频了
"""


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

def get_image(image_path):
    img_lists = find_data_list(image_path, suffix='jpg')
    mat = []
    for idx in trange(len(img_lists)-250,desc='正在将图像全部读如内存ing...',colour='GREEN'):
        frame_copy = cv2.imread(img_lists[idx])  #这一步骤就是把图全读进来了，别这样把
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
    img_shape = Images_list[0].shape
    width, height = img_shape[0], img_shape[1]

    # 创建视频编码器
    video_name = os.path.join(VideoSavePath,'output_video.mp4')
    # video_name = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要更改编码器（如：XVID、MJPG等）
    video_out = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    # 将每个图像逐帧写入视频
    for frame in range(len(Images_list)):
        video_out.write(Images_list[frame])

    # 释放资源
    video_out.release()
    print('视频已保存完成')

def trackmethod(inputvideo, original_position, savepath):
    txtpath = os.path.join(savepath,'txt')
    if not os.path.exists(txtpath):
        os.makedirs(txtpath)
    # txtpath2 = os.path.join(os.path.dirname(savepath), 'model_000')
    # if not os.path.exists(txtpath2):
    #     os.makedirs(txtpath2)

    outimgpath = os.path.join(savepath, 'output')
    if not os.path.exists(outimgpath):
        os.makedirs(outimgpath)

    mat3 = []
    row, col, p = inputvideo[0].shape
    t_start = time.time()
    for ii in range(len(original_position)):
        # print(name[ii])
        a = time.time()
        tracker = mecfTracker1()
        bbox = list(original_position[ii])
        # print(bbox)
        mat2 = []
        for mm in range(len(inputvideo)):
            if mm == 0:
                tracker.init(inputvideo[mm], bbox)
                mat2.append(bbox)
            else:
                x, y, w, h = bbox
                if x < 0 or y < 0 or x + w > col or y + h > row:
                    break
                # print(bbox)
                ok, bbox = tracker.update(inputvideo[mm])
                bbox = list(map(int, map(np.round, bbox)))
                mat2.append(bbox)
        b = time.time() - a
        print('FPS:',len(inputvideo)/b)
        txt = os.path.join(txtpath, str(ii+1).zfill(4)+'.txt')
        # txt2 = os.path.join(txtpath2, name[ii] + '.txt')
        np.savetxt(txt,np.array(mat2),fmt='%d',delimiter=',')
        # np.savetxt(txt2, np.array(mat2), fmt='%d', delimiter=',')
        mat3.append(mat2)
    t_end = time.time()
    print('All time: ', t_end - t_start)

    color = []
    outputvideo = copy.copy(inputvideo)
    for qq in range(len(mat3)):
        p = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        color.append(p)

    print(len(outputvideo))
    for jj in range(len(outputvideo)):
        for kk in range(len(mat3)):
            lent = len(mat3[kk])
            if lent <= jj:
                continue
            list2 = mat3[kk][jj]
            cv2.rectangle(outputvideo[jj], (list2[0], list2[1]), (list2[0] + list2[2], list2[1] + list2[3]), color[kk], 3)
        cv2.imwrite(os.path.join(outimgpath, str(jj+1).zfill(6)) + '.jpg', outputvideo[jj])


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
    
    # Get Memory Info
    process = psutil.Process()
    mem_info = process.memory_info()
    print (' --当前进程的内存使用：',psutil.Process(os.getpid()).memory_info().rss)
    print (' --当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

    # Generate Video File
    Images_to_Video(img_mat, result_path, 10)


