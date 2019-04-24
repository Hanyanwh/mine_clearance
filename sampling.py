import cv2
import win32gui
from PIL import ImageGrab
import numpy as np
import Neural_network as nn

"""
    检验训练结果
    
"""

# bp = nn.BPNetwork()
# bp.train()
HWND = win32gui.FindWindow(None, "扫雷")
rect = win32gui.GetWindowRect(HWND)

# 修正主窗口坐标main_box和状态窗口坐标
rect_main = (rect[0]+15, rect[1]+101, rect[2]-11, rect[3]-11)
main_box = rect_main[3] - rect_main[1]
rect_state = ((rect[0]+rect[2])/2-5, rect[1]+67, +(rect[0]+rect[2])/2+10, rect[3]-main_box-30)

pic_state = ImageGrab.grab(rect_state)
pic_main = ImageGrab.grab(rect_main)
pic = ImageGrab.grab(rect)
# 坐标main_box
state = pic_state
print(state)
img_main = cv2.cvtColor(np.asarray(pic_main), cv2.COLOR_RGB2GRAY)

img = cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2GRAY)

img_state = cv2.cvtColor(np.asarray(state), cv2.COLOR_RGB2GRAY)
circles_state, img_state = cv2.threshold(img_state, 130, 255, cv2.THRESH_BINARY_INV)
# print(bp.distinguish(img_state))
cv2.imshow('OpenCV', img_main)
cv2.imwrite("img.bmp", img_main)
cv2.waitKey()

# img = cv2.imread(".\\resource\\10.bmp")
# img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
# circles_state, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('open', img)
# cv2.waitKey()
