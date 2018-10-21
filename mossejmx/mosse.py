#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#jiameixia
import cv2
import numpy as np
import os



opts = {}

opts["the learning rate"]=0.125
opts["sigma"]=100
opts["num_pretrain"]=128
opts["rotate"]=False
opts["record"] ='store_true'
opts["video path"]='/home/meixia/Documents/论文阅读/moose/mosse-object-tracking/datasets/surfer'
    # return opts
def linear_mapping(images):
    # 线性化图像
    max_value = images.max()
    min_value = images.min()

    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a

    image_after_mapping = parameter_a * images + parameter_b

    return image_after_mapping

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

def pre_process(img):  #原始图像resize到和高斯一样大
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)   #e为底 每个像素点取log
    img = (img - np.mean(img)) / (np.std(img) + 1e-5) #标准差np.std
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img

def random_warp(img):
    a = -180 / 16   #a=-11.25
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()   #旋转角度范围是45度
    # 从一个均匀分布[low, high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    # 参数介绍:
    # low: 采样下界，float类型，默认值为0；
    # high: 采样上界，float类型，默认值为1；
    # size: 输出样本数目，为int或元组(tuple)
    # 类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个值。

    # rotate（翻转） the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1) #得到一个旋转矩阵
    #以图片中心旋转r的角度
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0])) #把原图像旋转了
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

def pre_training(init_frame, G):
    height, width = G.shape
    fi = cv2.resize(init_frame, (width, height))
    #图像预处理:取log，加窗
    fi = pre_process(fi)

    Ai = G * np.conjugate(np.fft.fft2(fi))  # 取共轭
    Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
    # for i in range(opts["num_pretrain"]):
    #     if opts["rotate"]:
    #         fi = pre_process(random_warp(fi))
    #     else:
    #         fi = pre_process(fi)
    #     Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
    #     Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

    return Ai, Bi


def get_gauss_response( img, gt):
    height, width = img.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    center_x = gt[0] + 0.5 * gt[2]
    center_y = gt[1] + 0.5 * gt[3]
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * opts["sigma"])
    response = np.exp(-dist)
    # normalize...
    #response = linear_mapping(response)
    return response

def tracking():
    #获取视频所有帧的名称列表
    img_path = opts["video path"]
    lines=os.listdir(img_path)
    for i in range(0, len(lines)):
        lines[i] = img_path +'/'+ lines[i]
    lines.sort()

    #对第一帧做操作
    #(1)感兴趣区域画框
    init_img = cv2.imread(lines[0])
    init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)  # 色彩空间转化函数
    init_frame = init_frame.astype(np.float32)
    init_gt = cv2.selectROI('demo', init_img, False, False)
    init_gt = np.array(init_gt).astype(np.int64)    #得到的框的区域
    #(2)将图片转换为二维矩阵
 # 原始图像的二维矩阵
    #(3)以感兴趣区域中心求图片的高斯函数
    response_map = get_gauss_response(init_frame, init_gt)     #图片的高斯响应
    g = response_map[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]    #从图片的高斯响应截取感兴趣区域的高斯响应
    G = np.fft.fft2(g)  #将感兴趣区域的高斯响应转换到频域
    fi = init_frame[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]     #截取图片的感兴趣区域部分
    Ai, Bi = pre_training(fi, G)
    for idx in range(len(lines)):
        current_frame = cv2.imread(lines[idx])
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype(np.float32)
        if idx == 0:
            Ai = opts["the learning rate"] * Ai
            Bi = opts["the learning rate"] * Bi
            pos = init_gt.copy()    #框的区域
            clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
        else:
            Hi = Ai / Bi
            fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
            Gi = Hi * np.fft.fft2(fi)   #从现在的图片上crop上一帧相同位置的图片，并且乘以滤波器，得到G
            gi = linear_mapping(np.fft.ifft2(Gi))
            # find the max pos...
            max_value = np.max(gi)    #找到现在这个图片的最大值
            max_pos = np.where(gi == max_value)     #找到最大值的位置
            # 计算现在要偏移多少
            dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
            dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

            # update the position...
            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy    #现在的图片中心位置

            # trying to get the clipped position [xmin, ymin, xmax, ymax]
            # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于
            # a_max，小于a_min, 的就使得它等于a_min。
            clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
            clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
            clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
            clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
            clip_pos = clip_pos.astype(np.int64)#左上右下

            # get the current fi..
            fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
            # online update...
            Ai = opts["the learning rate"] * (G * np.conjugate(np.fft.fft2(fi))) + (1 - opts["the learning rate"]) * Ai
            Bi = opts["the learning rate"] * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - opts["the learning rate"]) * Bi

        # visualize the tracking process...
        cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
        cv2.imshow('demo', current_frame)
        cv2.waitKey(1)


tracking()