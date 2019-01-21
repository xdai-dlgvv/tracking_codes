#预训练
import os
import cv2
import numpy as np
from others import linear_mapping
import pre_process


class mosse:
    def __init__(self, args, img_path):
        # 参数
        self.args = args
        self.img_path = img_path  # 图片路径
        # 得到视频中的每一帧
        self.frame_list = self._get_img_list(self.img_path)
        self.frame_list.sort()
        
    # 将视频中的每一帧图片存为图片列表
    def get_img_list(self, img_path):
        frame_list = []  # 建立一个空列表存放视频中的所有图片
        # 从第一帧开始将视频中的每帧存放到列表中
        for frame in os.listdir(img_path):
            # 只要后缀为jpg则存放到列表中
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame))
        # 返回图片列表
        return frame_list
    
    # 得到视频序列第一帧的groundtruth
    def get_int_ground_truth(self, img_path):
        # 将图片序列的路径转换为groundtruth的路径
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            line = f.redline()
            gt_pos = line.split(',')
        return [float(element) for element in gt_pos]
            
    # 训练第一帧
    def pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame,(width,height))
        fi = pre_process(fi)
        # 初始滤波器
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        # 仿射变换更新滤波器
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        return Ai, Bi

    # 高斯响应
    def gauss_response(self,img,gt):
        # 图片的高和宽
        height,width = img.shape
        # 将数组形式变为矩阵
        xx,yy = np.meshgrid(np.arange(width),np.arange(height))
        # 第一帧图片所画的ground truth的中心点坐标(center_x,center_y)
        center_x = gt[0] + 0.5*gt[2]
        center_y = gt[1] + 0.5*gt[3]
        # 高斯函数的指数
        dist = (np.square(xx - center_x) + np.square(yy - center_y))/(2 * self.args.sigma)
        # 高斯响应
        response = np.exp(-dist)
        # 数据规范化
        response = linear_mapping(response)
        return response
    
    # 进行跟踪
    def start_tracking(self):
        # 得到视频的第一帧图片
        # 读取第一帧，使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR
        # 注意是BGR，通道值默认范围0-255。
        init_img = cv2.imread(self.frame_list[0])
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)  # 彩图转换为灰色图
        init_frame = init_frame.astype(np.float32)
        # 画出第一帧的框
        init_gt = cv2.selectROI('demo', init_img, False, False)
        # 第一帧gt框左上角坐标和宽高
        init_gt = np.array(init_gt).astype(np.int64)
        # 得到第一帧的高斯响应图
        response_map = self._get_guass_response(init_frame, init_gt)
        # 开始创建跟踪序列
        # 高斯响应
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        # groundtruth的高斯响应图
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        # 进行傅里叶逆变换，得到原图的groundtruth
        G = np.fft.fft2(g)
        # 开始进行预训练，得到初始的Ai和Bi
        Ai, Bi = self._pre_training(fi, G)
        # 开始跟踪
        for idx in range(len(self.frame_lists)):
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            if idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                Gi = Hi * np.fft.fft2(fi)
                gi = linear_mapping(np.fft.ifft2(Gi))
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                # 新中心点相对于旧中心点的偏移
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                # online update...
                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi

                # visualize the tracking process...
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
            # 画框
            cv2.imshow('demo', current_frame)
            # 展示demo当前框
            cv2.waitKey(100)
            # if record... save the frames..
            if self.args.record:
                frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                if not os.path.exists(frame_path):
                    os.mkdir(frame_path)
                cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)



