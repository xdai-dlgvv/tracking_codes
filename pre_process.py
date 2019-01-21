#预处理
import numpy as np

def pre_process(img):
    #图片的高和宽
    height,width = img.shape
    #log：底数是e
    img = np.log(img+1)
    #归一化：z-score标准化
    x = np.mean(img)
    s = np.std(img)
    img = (img - x)/s
    #余弦窗：汉明窗
    window = hanning_cos(height,width)
    img = img * window
    return img

#hanning余弦窗
def hanning_cos(height,width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    #以win_col为行向量，win_row的维数为行数生成mask_col；
    # 以win_row为列向量，win_col为列数生成mask_row。
    mask_col,mask_row = np.meshgrid(win_col,win_row)
    win = mask_col * mask_row
    return win
