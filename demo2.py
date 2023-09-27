import os
from glob import glob
import numpy as np
import cv2
import random
import copy
import time
from mecfTracker1 import mecfTracker1
import cv2
import numpy as np
import os
from glob import glob
import random
import copy
from ATL_path import scandir, mkdir_or_exist, find_data_list

def gettxt(datapath):
    # path1 = glob(os.path.join(gtpath, '*'))
    # print(path1)
    name = os.listdir(datapath)
    mat = []
    for i in range(len(name)):
        print(name[i])
        txtpath = glob(os.path.join(datapath, name[i],'groundtruth_rect*'))
        gt = np.loadtxt(txtpath[0], dtype=int, delimiter=',')
        print(txtpath,':  ',list(gt[0]))
        mat.append(list(gt[0]))
    return mat,name

def get_image(path):
    files = os.listdir(path)
    files.sort()
    mat = []
    for filename in files:
        frame_copy = cv2.imread(os.path.join(path, filename))
        mat.append(frame_copy)
    return mat, files


def trackmethod(inputvideo, xx, savepath):
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
    for ii in range(len(xx)):
        # print(name[ii])
        a = time.time()
        tracker = mecfTracker1()
        bbox = list(xx[ii])
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

mat = [[4013,4395,13,13], [4317,3757,10,8], [1117,3595,38,13], [1239,3010,14,10], [2285,3297,21,12], [2634,3023,17,19], [3114,1904,25,31], [3109,2107,25,32],
       [3476,1555,19,12], [4000,556,38,64], [4785,671,11,17], [3913,3827,20,38], [3541,4046,16,12]]

imgpath = './JL1_1/'
resultpath = './JL1_1result/'
mkdir_or_exist(resultpath)
inputvideo, files = get_image(imgpath)
trackmethod(inputvideo, mat, resultpath)


# mat = [2,4,6,9,10,11,17]
# mat2 = [2,4,8,10]

# for i in range(len(mat2)):
#     name = 'car_'+str(mat2[i]).zfill(3)
#     datapath = os.path.join('D:\\BaiduNetdiskDownload\\sot\\data2', name)
#     resultpath = os.path.join('D:\\BaiduNetdiskDownload\\sot\\occlusionresult\\dat', name)
#     inputvideo,files = get_image(os.path.join(datapath,os.path.basename(datapath)+'_1','img'))
#     xx,name = gettxt(datapath)
#     trackmethod(inputvideo, xx, name, resultpath)
