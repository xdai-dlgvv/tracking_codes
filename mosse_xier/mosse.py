#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 18-10-17 下午4:00 
# @Author : Xier
# @File : mosse.py

import os
import cv2
import numpy as np
import random


# mosse算法的主体部分
'''
_get_gt(self)
_normalize(self, img)
_guass_response(self, img, gt)
_apply_hanning(self, img)
_affine(self,img)
_pre_training(self, ori_img, G0)
'''
class Mosse:
    def __init__(self, img_path, gt_path):
        self.lr = 0.125  # 学习率
        self.sigma = 100  # 生成的高斯响应图的方差
        self.affine = False  # 是否在第一帧训练过程中加入仿射变换
        self.num_train = 8  # 生成的仿射变换图片的数量
        self.img_path = img_path   # 图片路径
        self.gt_path = gt_path  # 目标框的位置


    # 作用：获得物体的groundtruth
    # 输入：txt
    # 输出：groundtruth
    def _get_gt(self):
        with open(self.gt_path, 'r') as f:
            line = f.read()
            gt_pos = line.split('\n')[0].split(',')
        return [int(element) for element in gt_pos]


    # 作用：线性归一化
    # 输入：图片
    # 输出：归一化后的图片
    def _normalize(self, img):

        # 获取图片像素点的最小值和最大值
        min = img.min()
        max = img.max()

        # 对图片进行归一化
        img_normal = (img - min)/(max - min)

        return img_normal


    # 作用：生成高斯响应图
    # 输入：第一帧根据目标框切图，groundtruth box 的坐标信息（左上x，左上y，宽，高）
    # 输出：该切图的高斯响应图
    def _guass_response(self, img, gt):

        h, w = img.shape

        # 按图片的宽和高，生成量化的坐标矩阵
        mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))

        # 找到目标中心点坐标信息
        center_x = gt[2] / 2
        center_y = gt[3]/2

        # 计算高斯响应
        dist = (np.square(mesh_x-center_x)+np.square(mesh_y-center_y))/(2*self.sigma)
        response = np.exp(-dist)

        # 对生成的高斯响应图进行归一化
        response = self._normalize(response)

        return response


    # 作用：对图片加hanning窗，减小图片对比度和边缘效应
    # 输入：图片
    # 输出：加hanning窗之后的图片
    def _apply_hanning(self, img):

        h,w = img.shape

        # 压缩图片的像素值，减小图片的对比度，并化为标准分布
        img = np.log(img+1)
        img = (img-np.mean(img))/(np.std(img)+ 1e-5)

        # 将生成的宽方向和高方向的hanning窗相乘，生成二维的hanning窗
        mask_col, mask_row = np.meshgrid(np.hanning(w), np.hanning(h))
        win = mask_col*mask_row

        # 在图片上应用hanning窗
        img = img*win

        return img


    # 作用：仿射变换
    # 输入：第一帧目标框切图
    # 输出：随机仿射后的图片
    def _affine(self,img):

        h, w = img.shape

        # 选择图片上的三个像素点坐标
        input_pts = np.float32([[0,0],[w-1, 0],[0, h-1]])

        # 随机生成三个像素坐标经过仿射变换后的坐标
        output_pts0 = np.array([int(w*random.uniform(0, 0.5)/10),int(h*random.uniform(0, 0.5)/10)])
        output_pts1 = np.array([int(w * random.uniform(9.5, 10) / 10), int(h * random.uniform(0, 0.5) / 10)])
        output_pts2 = np.array([int(w * random.uniform(0, 0.5) / 10), int(h * random.uniform(9.5, 10) / 10)])
        output_pts = np.float32([output_pts0, output_pts1, output_pts2])

        # 执行仿射变换
        M = cv2.getAffineTransform(input_pts, output_pts)
        img_aff = cv2.warpAffine(img, M, (w, h))

        return img_aff


    # 作用：对第一帧图片进行仿射变换并训练
    # 输入：第一帧目标框切图，该目标框切图的高斯响应图
    # 输出：训练出的滤波器Hi的分子Ai,分母Bi
    def _pre_training(self, ori_img, G0):

        # 对第一帧图像加hanning窗
        fi = self._apply_hanning(ori_img)

        # 产生Ai和Bi
        Ai = G0 *np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(fi)*np.conjugate(np.fft.fft2(fi))


        # 若self.affine为true，进行仿射变换
        for i in range(self.num_train):
            if self.affine:
                fi = self._apply_hanning(self._affine(ori_img))
            else:
                fi = self._apply_hanning(ori_img)


            # 累加Ai和Bi
            Ai = Ai + G0 *np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))


        return Ai, Bi


# 作用：跟踪代码主体部分
# 输入：图片路径
# 输出：视频每一帧目标的位置信息（左上x，左上y，宽，高）
def perfrom_tracking(Mosse):

    # 读取图片文件夹下的所有图片名并排序
    img_list = os.listdir(Mosse.img_path)
    img_list.sort()

    # 读入第一张图片并转化为灰度图像
    first_frame = cv2.imread(os.path.join(Mosse.img_path, img_list[0]))
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = first_gray.astype(np.float32)


    # 读取grountruth box 的信息（左上x，左上y，宽，高）
    first_gt = Mosse._get_gt()

    # 根据grountruth box切取灰度图中的目标位置
    crop_frame = first_gray[first_gt[1]:first_gt[1]+first_gt[3],first_gt[0]:first_gt[0]+first_gt[2]]

    # 求目标区域的高斯响应
    g = Mosse._guass_response(crop_frame, first_gt)
    G0 = np.fft.fft2(g)

    # 进行仿射变换训练第一张图像的Ai，Bi，即滤波器的分子分母
    Ai, Bi = Mosse._pre_training(crop_frame, G0)

    box = first_gt.copy()


    for idx in range(len(img_list)):

        # 读入当前图片并转化为灰度图像
        current_frame = cv2.imread(os.path.join(Mosse.img_path, img_list[idx]))
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_gray = current_gray.astype(np.float32)

        # 开始跟踪过程
        if idx == 0:
            tracker_list = [box.copy()]

            # clip_pos为在图上画框的位置
            clip_pos = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])

        else:
            if idx == 1:

                # 在当前图上目标框区域进行切图，并加hanning窗
                fi = current_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = Mosse._apply_hanning(cv2.resize(fi, (box[2], box[3])))

            # 生成滤波器Hi
            Hi = Ai / Bi

            # 根据滤波器Hi和目标区域图Fi生成响应图Gi
            Gi = Hi*np.fft.fft2(fi)
            gi = np.fft.ifft2(Gi)
            gi = Mosse._normalize(gi)

            # 展示响应图
            cv2.imshow('g', np.real(gi))
            cv2.waitKey(10)

            # 找到最高响应的位置
            max_pos = np.where(gi == np.max(gi))

            # 计算最高响应位置的偏移量
            dx = int(np.mean(max_pos[1]) - box[2]/2)
            dy = int(np.mean(max_pos[0]) - box[3]/2)

            # 修正目标框的位置
            box[0] = box[0] + dx
            box[1] = box[1] + dy

            # 限制目标在图上画框的位置不超出图像边缘
            clip_pos[0] = np.clip(box[0], 0, current_frame.shape[1])
            clip_pos[1] = np.clip(box[1], 0, current_frame.shape[0])
            clip_pos[2] = np.clip(box[0]+box[2], 0, current_frame.shape[1])
            clip_pos[3] = np.clip(box[1]+box[3], 0, current_frame.shape[1])

            # 在当前图上目标框区域进行切图，并加hanning窗
            fi = current_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
            fi = Mosse._apply_hanning(cv2.resize(fi, (box[2], box[3])))
            Fi = np.fft.fft2(fi)

            # 更新Ai和Bi
            Ai = Mosse.lr * G0 * np.conjugate(Fi) + (1 - Mosse.lr) * Ai
            Bi = Mosse.lr * Fi * np.conjugate(Fi) + (1 - Mosse.lr) * Bi

            # 记录每一帧目标位置
            tracker_list.append(box.copy())

        # 在原图上画框
        cv2.rectangle(current_frame,(clip_pos[0], clip_pos[1]), (clip_pos[2], clip_pos[3]),(255, 0, 0),2)
        cv2.imshow('tracker', current_frame)
        cv2.waitKey(10)

    return tracker_list


if __name__ == '__main__':

    # 输入图片路径和groundtruth
    img_path = '/home/chenxier/mosse-object-tracking/datasets/surfer'
    gt_path = 'groundtruth.txt'

    # 进行跟踪
    tracker = Mosse(img_path, gt_path)
    tracker_list = perfrom_tracking(tracker)

    # 将结果记录进txt文件
    with open('result.txt','w') as f:
        for i in range(len(tracker_list)):
            s = str(tracker_list[i]).replace('[', '').replace(']','')
            s = s + '\n'
            f.write(s)
