import os
import cv2
import numpy as np
import argparse
from utils import *

class mosse:
    def __init__(self,args,img_path):
        self.args = args
        self.img_path = img_path
        self.frame_lists = self._get_img_lists(self.img_path)
        
    def start_tracking(self):
        #cv2.imread()接口读图像，BGR 格式 0~255，通道格式为(W,H,C)
        init_img = cv2.imread(self.frame_lists[0])
        #cv2.cvtColor()颜色转换
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        #np.astype()强制类型转换
        init_frame = init_frame.astype(np.float32)
        #tuple（gt左上角坐标、宽、高）=cv2.selectROI()选择感兴趣的区域gt
        init_gt = cv2.selectROI('demo', init_img, False, False)
        #将tuple转换成array格式
        init_gt = np.array(init_gt).astype(np.int64)

        response_map = self._get_gauss_response(init_frame, init_gt)
        #框出gt在高斯响应图的区域
        g = response_map[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
        # 框出gt在原图的区域
        fi = init_frame[init_gt[1]:init_gt[1] + init_gt[3], init_gt[0]:init_gt[0] + init_gt[2]]
        #将高斯图转到频域
        G = np.fft.fft2(g)

        #预训练
        Ai, Bi = self._pre_training(fi, G)

        #开始追踪
        for idx in range(len(self.frame_lists)):
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            if idx == 0:
                # Ai = self.args.lr * Ai
                # Bi = self.args.lr * Bi
                Ai = Ai
                Bi = Bi
                #复制gt（gt左上角坐标、宽、高）
                pos = init_gt.copy()
                #pos(左上角坐标、右下角坐标）
                clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                #得到第一帧gt在第二帧灰度区域的fi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                #算出第二帧的gi
                Gi = Hi * np.fft.fft2(fi)
                gi = linear_mapping(np.fft.ifft2(Gi))
                # find the max pos...
                max_value = np.max(gi)
                #最高点相对于gi左上角的位置
                max_pos = np.where(gi == max_value)

                #新中心点相对于旧中心点的偏差
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

                # update the position...
                #第二帧gt左上角坐标
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                # trying to get the clipped position [xmin,
                # ymin, xmax, ymax]
                #得到第二帧gt的坐标（左上右下）
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # get the current fi..
                #得到第二帧gt在第二帧灰度区域的fi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi

            # visualize the tracking process...
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame)
            cv2.waitKey(100)
            # if record... save the frames..
            # if self.args.record:
            #     frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
            #     if not os.path.exists(frame_path):
            #         os.mkdir(frame_path)
            #     cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)

    def _get_img_lists(self, img_path):
        frame_list = []
        #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for frame in os.listdir(img_path):
            #os.path.splitext(path) 分割路径，返回路径名0001和文件扩展名.jpg的元组
            if os.path.splitext(frame)[1] == '.jpg':
                #os.path.join(path1[, path2[, ...]])把目录和文件名合成一个路径
                frame_list.append(os.path.join(img_path, frame))
        #sort()函数用于对原列表进行排序
        frame_list.sort()
        return frame_list

    def _get_gauss_response(self, img, gt):
        height, width = img.shape
        #np.meshgrid()生成网格型数据  np.arange()函数返回一个有终点和起点的固定步长的数组
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # 目标的中心点坐标
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        #高斯函数
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * 100)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    def _pre_training(self, init_frame, G):
        height, width = G.shape
        #将gt的原图和高斯响应图 尺寸统一了？ 明明一样
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        #  预处理 先log 4. 5.  z-score归一化 0-1  再二维余弦（汉明）弱化边缘
        fi = pre_process(fi)

        Ai = G * np.conjugate(np.fft.fft2(fi))

        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        #仿射变换 128
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

        return Ai, Bi



#argparse.ArgumentParser()创建一个命令解析器的句柄
parse = argparse.ArgumentParser()
#parse.add_argument()添加命令解析器命令,读入命令行参数
parse.add_argument('-lr', type=float, default=0.125, help='the learning rate')
parse.add_argument('-sigma', type=float, default=100, help='the sigma')
parse.add_argument('-num_pretrain', type=int, default=128, help='the number of pretrain')
#action="store_true" 表示该选项不需要接收参数
parse.add_argument('-rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('-record', action='store_true', help='record the frames')


if __name__ == '__main__':
    #parse_args()方法进行解析
    args = parse.parse_args()
    img_path = '/home/lixiaoxue/mosse-object-tracking/datasets/surfer/'
    tracker = mosse(args, img_path)
    tracker.start_tracking()
