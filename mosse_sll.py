#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : lynn

import numpy as np
import cv2
import os

learning_rate = 0.125 #学习率
sigma = 100  # 生成的高斯响应图的方差
num_pretrain = 32 # 生成的仿射变换图片的数量


def linear_mapping(images):

    '''
    作用：线性化图像
    :param images: 输入图片
    :return: 线性化归一化后的图片
    '''

    # 获取图片像素点的最小值和最大值
    max_value = images.max()
    min_value = images.min()

    #对图片进行归一化
    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a
    image_after_mapping = parameter_a * images + parameter_b
    return image_after_mapping

def window_haning(height, width):

    '''
    作用：对图片加hanning窗，减小图片对比度和边缘效应
    :param height: 图片高度
    :param width: 图片宽度
    :return: 加hanning窗之后的图片矩阵
    '''

    win_col = np.hanning(width)
    win_row = np.hanning(height)

    # 将生成的宽方向和高方向的hanning窗相乘，生成二维的hanning窗
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row #矩阵元素点乘, np.dot()矩阵相乘
    return win

def pre_process(img):

    '''
    作用：预处理－压缩图片的像素值，减小图片的对比度，并化为标准分布
    :param img: 输入图片
    :return: 处理后图片
    '''

    # 获取图片大小
    height, width = img.shape

    #压缩图片的像素值，减小图片的对比度，并化为标准分布
    img = np.log(img + 1)   #log()表示e为底,log10()10为底，每个像素点取log
    img = (img - np.mean(img)) / (np.std(img) + 1e-5) #标准差np.std

    # 使用汉宁窗
    window = window_haning(height, width)
    img = img * window
    return img

def pre_training(init_frame, G):

    '''
    作用：预训练
    :param init_frame: 第一帧图像目标框切图
    :param G: 目标框切图的高斯响应图
    :return: 训练出的滤波器Hi的分子Ai,分母Bi
    '''

    height, width = G.shape
    fi = cv2.resize(init_frame, (width, height))

    #图像预处理,取log,加窗
    fi = pre_process(fi)
    Ai = G * np.conjugate(np.fft.fft2(fi))  # 取共轭
    Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
    return Ai, Bi

def get_gauss_response( img, gt):

    '''
    作用：对图像进行高斯滤波
    :param img: 第一帧根据目标框切图
    :param gt: groundtruth的坐标信息（左上x,左上y,宽,高）
    :return: 高斯响应图
    '''

    #获取图片宽高
    height, width = img.shape

    # 按图片的宽和高，生成量化的坐标矩阵
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # 找到目标中心点坐标信息
    center_x = gt[0] + 0.5 * gt[2]
    center_y = gt[1] + 0.5 * gt[3]

    # 计算高斯响应
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * sigma)
    response = np.exp(-dist)
    return response

def mosse_tracking(img_path, gt_txt):

    '''
    作用：实现跟踪
    :param img_path: 输入图片路径
    :param gt_txt: 标注框信息文件
    :return:
    '''
    #获取视频所有帧的名称列表
    lines=os.listdir(img_path)
    for i in range(0, len(lines)):
        lines[i] = img_path +'/'+ lines[i]
    lines.sort()

    #第一帧感兴趣区域画框
    init_img = cv2.imread(lines[0])
    init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    init_frame = init_frame.astype(np.float32)
    init_gt = cv2.selectROI('demo', init_img, False, False)
    init_gt = np.array(init_gt).astype(np.int64)

    #以感兴趣区域中心求图片的高斯函数
    response_map = get_gauss_response(init_frame, init_gt)
    g = response_map[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
    G = np.fft.fft2(g)
    fi = init_frame[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
    Ai, Bi = pre_training(fi, G)
    for idx in range(len(lines)):
        current_frame = cv2.imread(lines[idx])
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype(np.float32)
        if idx == 0:
            Ai = learning_rate * Ai
            Bi = learning_rate * Bi
            pos = init_gt.copy()
            clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
        else:
            Hi = Ai / Bi
            fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

            # 从现在的图片上裁剪上一帧相同位置的图片，并且乘以滤波器，得到Gｉ
            Gi = Hi * np.fft.fft2(fi)
            gi = linear_mapping(np.fft.ifft2(Gi))

            # 找到当前图片的最大值
            max_value = np.max(gi)
            max_pos = np.where(gi == max_value)

            # 计算偏移量
            dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
            dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

            # 更新图片位置
            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy

            # 获取点击位置 [xmin, ymin, xmax, ymax]
            clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
            clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
            clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
            clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
            clip_pos = clip_pos.astype(np.int64)

            # 得到目前fi
            fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

            # 在线更新Ａｉ,B i
            Ai = learning_rate * (G * np.conjugate(np.fft.fft2(fi))) + (1 - learning_rate) * Ai
            Bi = learning_rate * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - learning_rate) * Bi

        # 追踪过程可视化
        xx = str(pos[0]) + ',' + str(pos[1]) + ',' + str(pos[0] + pos[2]) + ',' + str(pos[1] + pos[3])

        if not os.path.exists(gt_txt):
            os.mknod(gt_txt)
        f = open(gt_txt, 'a')
        f.write(xx + '\n')
        cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
        cv2.imshow('demo', current_frame)
        cv2.waitKey(1)
        # return


if __name__ == '__main__':
    img_path = 'datasets/surfer/'  # 图片路径
    gt_txt = 'groundtruth.txt'  # 目标框的位置
    mosse_tracking(img_path, gt_txt)


