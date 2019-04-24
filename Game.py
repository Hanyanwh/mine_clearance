import cv2
import win32gui
from ctypes import windll
import win32con
import win32api
from PIL import ImageGrab
import numpy as np
import time
import Neural_network as nn
"""
    思路：
        1. 将扫雷窗口解释为矩阵
        2. 进行矩阵运算推测雷的位置
        3. 鼠标模拟点击
        
        因为懒得收集训练数据，只是用少量数据训练模型，所以BP神经网络过拟合严重，
        如果运行后出现异常，多半是识别出错了，请重新收集数据，同时使用Neural_network.py生成模型
        
    文件说明：
        1~12.bmp: 是训练数据，对应数字，状态，小旗
        Data.mat: 储存BP神经网络的权值矩阵，当存在该文件时会自动加载权值，
                    否则将进行训练
        Game.py: 扫雷主脚本，包含扫雷算法，模拟点击算法，猜测算法
        Neural_network: BP神经网络脚本，用于模式识别
"""

NULL = -2


def mineClearance():
    # 创建识别对象
    recognition = nn.BPNetwork()
    recognition.pretreatment()
    recognition.train()

    # 调用windowAPI获取扫雷窗口句柄和窗口坐标
    HWND = win32gui.FindWindow(None, "扫雷")
    rect = win32gui.GetWindowRect(HWND)

    # 主窗口坐标和状态窗口坐标
    rect_main = (rect[0] + 15, rect[1] + 101, rect[2] - 11, rect[3] - 11)
    main_box = rect_main[3] - rect_main[1]
    rect_state = ((rect[0] + rect[2]) / 2 - 5, rect[1] + 67, +(rect[0] + rect[2]) / 2 + 10, rect[3] - main_box - 30)
    rect_mouse = (rect_main[0], rect_main[1], rect_main[2], rect_main[3])
    time.sleep(0.5)
    length = int((rect_main[2] - rect_main[0])/16)
    high = int((rect_main[3] - rect_main[1])/16)
    # 首先点击四个角如果有雷就重开。。。
    initialization = np.array([[1, 1],
                               [1, 1],
                               [length, 1],
                               [1, high],
                               [length, high]])
    mouseEven([], initialization, rect_mouse)

    flag = 1
    times = 0
    while flag == 1:
        # 迭代次数
        times += 1
        # 抓图
        pic_state = ImageGrab.grab(rect_state)
        # 分块截图
        state = pic_state  # 状态
        img_state = cv2.cvtColor(np.asarray(state), cv2.COLOR_RGB2GRAY)
        circles_state, img_state = cv2.threshold(img_state, 120, 255, cv2.THRESH_BINARY_INV)

        state = recognition.distinguish(img_state)
        if state == 9:
            # 9对应的状态是笑脸，也就是正常
            pass
        elif state == 12:
            # 12对应的是有墨镜的状态，也就是完成比赛

            # 打印心情
            print("*\( ^ v ^ )/*")
            break
        elif state == 10:
            # 10对应的是失败

            # 打印心情
            print("≡(▔﹏▔)≡")
            break
        else:
            break
        main_mat = np.zeros([high + 2, length + 2])
        pic = ImageGrab.grab(rect_main)

        for i in range(high):
            for o in range(length):
                # 获取每一块的图片
                main = pic.crop((o * 16+1, i * 16+1, o * 16 + 16, i * 16 + 16))

                # 转化为灰度图
                img = cv2.cvtColor(np.asarray(main), cv2.COLOR_RGB2GRAY)

                if np.sum(img == 255) >= 10 and np.sum(img == 0) == 0:
                    # 判断块是否为未点击
                    main_mat[i + 1, o + 1] = NULL
                elif np.sum(img == 192) >= 250:
                    # 判断块是否已经被点击且无数字
                    main_mat[i + 1, o + 1] = 0
                else:
                    # 解析数字
                    # 二值化
                    circles, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
                    num = recognition.distinguish(img)
                    if num == 11:
                        # 如果为旗子则解释为-3
                        main_mat[i + 1, o + 1] = -3
                    else:
                        main_mat[i + 1, o + 1] = num

        # 调用分析方法，分析出安全点和危险点
        coordinate_danger, coordinate_safety = coreAlgorithm(main_mat)
        # 模拟鼠标点击
        mouseEven(coordinate_danger, coordinate_safety, rect_mouse)


def mouseEven(coordinate_danger, coordinate_safety, rect_mouse):
    # 模拟鼠标点击 (鼠标坐标与与真实坐标不同，鼠标坐标为真实值的0.8)
    # flag = 0 左击
    # flag = 1 右击
    for flag in range(2):
        if flag == 0 and len(coordinate_safety) != 0:
            # 危险坐标 右击
            coordinate = coordinate_safety
        elif flag == 1 and len(coordinate_danger) != 0:
            # 安全坐标 左击
            coordinate = coordinate_danger
        else:
            # 为空则跳出循环
            continue

        # 将相对坐标转化为绝对坐标
        coordinate = 16 * (coordinate - 1) + np.array([rect_mouse[0], rect_mouse[1]]).reshape(1, 2) + 8
        for i in range(len(coordinate)):
            # 初始化 x, y
            x, y = coordinate[i]
            x = int(x)
            y = int(y)
            # 移动鼠标至 x, y
            windll.user32.SetCursorPos(x, y)
            if flag == 0:
                # 左击
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif flag == 1:
                # 右击
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            # 休息一下
            time.sleep(0.01)


def coreAlgorithm(main_mat):
    # 扫雷核心算法

    # 初始化权值矩阵
    P = np.ones(shape=main_mat.shape) * -1
    # 初始化危险坐标矩阵
    output_danger = np.zeros(shape=[0, 2])
    # 初始化安全坐标矩阵
    output_safety = np.zeros(shape=[0, 2])
    # 获取数字的坐标
    index = np.argwhere(main_mat >= 1)

    # 对每个数字进行迭代
    for i in range(len(index)):
        # 获取坐标
        x, y = index[i]
        # 初始化雷数
        thunder = main_mat[x, y]
        # 初始化未点击块个数
        total = 8.0
        # 初始化未点击块的坐标矩阵
        null = np.zeros(shape=(2, 0))

        # 对每个坐标的周围位置进行迭代
        for r in range(x-1, x+2):
            for c in range(y-1, y+2):
                # 去除中心点
                if r == x and c == y:
                    continue
                # 如果为数字或者为空则，总未点击块个数减一
                if main_mat[r, c] == 0 or main_mat[r, c] >= 1:
                    total -= 1
                # 如果是未点击块则加入null，且初始化该位置的概率为0
                elif main_mat[r, c] == NULL:
                    null = np.column_stack((null, [r, c]))
                    if P[r, c] == -1:
                        P[r, c] = 0
                # 如果该块为小旗，则总块数减一，雷数减一
                elif main_mat[r, c] == -3:
                    total -= 1
                    thunder -= 1

        if total == 0:
            # 判断空块数是否为零
            continue
        else:
            # 计算权重
            p_thunder = thunder/total
        # 累加权重
        P[list(null.astype(int))] += p_thunder
        if p_thunder == 0:
            # 如果权重为0，则为安全块
            for t in range(len(null[0, :])):
                # 去重加入结果
                if null[:, t].reshape(1, 2) not in output_safety:
                    output_safety = np.row_stack((output_safety, null[:, t]))
        elif p_thunder == 1:
            # 如果权重为1，则为危险块
            for t in range(len(null[0, :])):
                # 去重加入结果
                if null[:, t].reshape(1, 2) not in output_danger:
                    output_danger = np.row_stack((output_danger, null[:, t]))
    # 判断危险块和安全块的数量是否为零
    # 若不为零则交换行列坐标
    if len(output_danger) != 0:
        output_danger = np.array([output_danger[:, 1], output_danger[:, 0]]).T
    if len(output_safety) != 0:
        output_safety = np.array([output_safety[:, 1], output_safety[:, 0]]).T
    # 若都为空则进行guess算法
    if len(output_danger) == 0 and len(output_safety) == 0:
        output_danger, output_safety = guess(P)
    return output_danger, output_safety


def guess(P):
    # 猜雷
    if len(np.argwhere(P == min(P[P > 0]))) > len(np.argwhere(P == max(P[P > 0]))):
        # 若果最小权重的个数大于最大权重的个数则取最大权重处为危险块
        output_danger = np.argwhere(P == max(P[P > 0]))
        output_danger = np.array([output_danger[0, 1], output_danger[0, 0]])
        output_safety = []
    else:
        # 相反
        output_danger = []
        output_safety = np.argwhere(P == min(P[P > 0]))
        output_safety = np.array([output_safety[0, 1], output_safety[0, 0]])

    # 打印计算机的心情
    print("(#‵′)凸")
    return output_danger, output_safety


if __name__ == '__main__':
    mineClearance()
