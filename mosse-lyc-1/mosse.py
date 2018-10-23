import numpy as np
import cv2
import csv
from until import *
import os
import argparse
class Mosse: #定义一个类,名为Mosse
    def __init__(self,args):
        self.args = args
        self.img_path = self.args.image_path
        self.img_list = self.get_img_lists(self.img_path)
        self.save_path = self.args.save_path
    def track(self):
        pool = [0.85,0.90,0.95,1,1.05,1.1,1.15]
        init_image = cv2.imread(self.img_list[0])
        init_frame = cv2.cvtColor(init_image,cv2.COLOR_BGR2GRAY)#先将图片转为灰度图
        init_frame = init_frame.astype(np.float32)
        gt = cv2.selectROI('mosse', init_image, False, False)
        #gt = np.array([242,132,114,155])
        gt = np.array(gt).astype(np.int64)
        gauss_img = gauss(init_frame,gt,self.args.sigma)#进行高斯操作
        init_gt = gauss_img[gt[1]:gt[1]+gt[3],gt[0]:gt[0]+gt[2]]
        init_gt_show = np.uint8(init_gt * 255)
        cv2.imshow('1', init_gt_show)#显示gi的高斯图
        cv2.waitKey(100)
        f = init_frame[gt[1]:gt[1]+gt[3],gt[0]:gt[0]+gt[2]]
        Ai, Bi = self.pro_training( f ,init_gt)

        if os.path.exists(os.path.join(self.save_path, 'box.txt')) == True:
            os.remove(os.path.join(self.save_path, 'box.txt'))

        for i in range(len(self.img_list)):
            image = cv2.imread(self.img_list[i])
            image_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image_frame = image_frame.astype(np.float32)
            if i == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                plot = gt.copy()
                clip = np.array([plot[0],plot[1],plot[0]+plot[2],plot[1]+plot[3]]).astype(np.int64)
                b = 3
            else:
                Hi = Ai / Bi
                #f = image_frame[clip[1]:clip[3],clip[0]:clip[2]]
                Gi = np.zeros(len(pool))
                for i in range(len(pool)):
                    h ,w =image_frame.shape
                    w1 = ((pool[i]-1)/2)*w
                    h1 = ((pool[i]-1)/2)*h

                    new_img = cv2.resize(image_frame,(int(w*pool[i]),int(h*pool[i])))
                    f = new_img[clip[1]+int(((pool[i]-1)/2)*w):clip[3]+int(((pool[i]-1)/2)*w),clip[0]+int(((pool[i]-1)/2)*h):clip[2]+int(((pool[i]-1)/2)*h)]
                    f = preprocessing(f)
                    Gii = np.fft.fft2(f) * (Hi)
                    gii = (np.fft.ifft2(Gii))
                    Gi[i] = np.max(gii)
                Gi = np.array(Gi)
                b = np.where(Gi == np.max(Gi))
                b = pool[b[0][0]]
                f = image_frame[clip[1]:clip[3], clip[0]:clip[2]]
                f = cv2.resize(f, (gt[2], gt[3]))
                f = preprocessing(f)
                G = np.fft.fft2(f)*(Hi)
                gi = normal(np.fft.ifft2(G))
                a = gi.copy()
                a = np.uint8(a * 255)
                cv2.imshow('1',a)#显示预测的gi的图片

                g_max = np.max(gi)
                g_plot = np.where(gi == g_max)
                dy = int(np.mean(g_plot[0]) - gi.shape[0]/2)
                dx = int(np.mean(g_plot[1]) - gi.shape[1]/2)

                plot[0] = plot[0] + dx
                plot[1] = plot[1] + dy
                clip[0] = np.clip(plot[0],0,image.shape[1])#保证fi都在图片内,重新调整fi的范围
                clip[1] = np.clip(plot[1], 0, image.shape[0])
                clip[2] = np.clip(plot[0]+plot[2], 0, image.shape[1])
                clip[3] = np.clip(plot[1]+plot[3], 0, image.shape[0])
                clip = clip.astype(np.int64)

                f = image_frame[clip[1]:clip[3],clip[0]:clip[2]]
                try:
                    f = cv2.resize(f, (gt[2], gt[3]))
                    f = preprocessing(f)
                    Ai = self.args.lr * (np.fft.fft2(init_gt) * np.conjugate(np.fft.fft2(f))) + (1 - self.args.lr) * Ai#更新Ai与Bi
                    Bi = self.args.lr * (np.fft.fft2(f) * np.conjugate(np.fft.fft2(f))) + (1 - self.args.lr) * Bi
                except:
                    print('object disappear!')
                    break
            aa = int(plot[0]+plot[2]*(b+1)/2)
            bb = int(plot[1]+plot[3]*(b+1)/2)
            cv2.rectangle(image,(int(plot[0]+plot[2]*(b-1)/2),int(plot[1]+plot[3]*(b-1)/2)),(aa,bb),(255, 0, 0), 2)
            cv2.imshow('mosse',image)
            cv2.waitKey(1)
            plot1 = np.array((plot[0],plot[1],plot[0]+plot[2],plot[1]+plot[3]))
            with open(self.save_path + 'box.txt','a') as file:#把框变化信息写入到txt中
                for j in range(len(plot1)):
                    data_plot = str(plot1[j])
                    if j < (len(plot1)-1):
                        file.write(data_plot+',')
                    else :
                        file.write(data_plot+'\t\n')
    def pro_training(self, f ,g):#预处理
        height, weight = g.shape
        f1 = cv2.resize(f, (weight,height))
        f1 = preprocessing(f1)
        F1 = np.fft.fft2(f1)
        G = np.fft.fft2(g)
        Ai = G*np.conjugate(F1)
        Bi = np.fft.fft2(f1)*np.conjugate(np.fft.fft2(f1))
        for _ in range(self.args.number_xuanzhuang):
            f1= preprocessing(random_xuanzhuang(f))
            Ai = Ai + G* np.conjugate(np.fft.fft2(f1))
            Bi = Bi + np.fft.fft2(f1)* np.conjugate(np.fft.fft2(f1))
        return Ai,Bi


    def get_img_lists(self,img_path):#读取所有图片帧的路径
        img_list = []
        images_path = os.listdir(img_path)
        for image in images_path:
            if os.path.splitext(image)[1] == '.jpg':
                img_list.append(os.path.join(img_path,image))
        img_list.sort()
        return img_list

if __name__== '__main__':
    parse = argparse.ArgumentParser()#使用argparse模块存放变量信息
    parse.add_argument('--sigma',type=float, default=10 ,help='the sigma')
    parse.add_argument('--lr',type=float,default=0.125,help='the learning rate')
    parse.add_argument('--number_xuanzhuang',type=float,default=128,help='the number of pretrain')
    parse.add_argument('--image_path', type=str, default='/home/lyc/mosse-object-tracking/datasets/surfer', help='path of image')
    parse.add_argument('--save_path', type=str, default='/home/lyc/PycharmProjects/moose-lyc/box/', help='save path')
    args = parse.parse_args()
    tracker = Mosse(args)
    tracker.track()