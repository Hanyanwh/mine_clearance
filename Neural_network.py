import numpy as np
import cv2
import scipy.io as so
"""
    BP神经网络图像识别
    
"""


class BPNetwork:
    def __init__(self):
        # 输入层神经元
        self.input_data = np.zeros([12, 15*15])
        # 输出层神经元
        self.output_data = np.zeros([12, 12])
        # 隐层输出层权值
        self.w = -1/np.sqrt(501) + 2/np.sqrt(501) * np.random.rand(501, 12)
        # 输入层隐层权值
        self.v = -1/np.sqrt(15*15+1) + 2/np.sqrt(15*15+1) * np.random.rand(15*15+1, 500)
        # 定义激活函数（S型函数）
        self.f = lambda x: 1/(1 + np.exp(-x))
        # 学习速率
        self.k = 0.05

    def pretreatment(self):
        # 图片预处理方法

        for i in range(12):
            # 导入训练图片
            img = cv2.imread(".\\resource\\"+str(i+1)+".bmp")
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            circles_state, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
            # 将图片转化为向量
            img_data = np.array(img).reshape(1, 15*15)
            # 二进制化
            img_data[img_data > 0] = 1
            # 更新训练数据
            self.input_data[i, :] = img_data
            self.output_data[i, i] = 1
        print("图片预处理完成！")

    def train(self):
        try:
            data = so.loadmat('Data.mat')
        except:
            # 预处理图片
            self.pretreatment()
        else:
            # 载入权值
            self.w = data['w']
            self.v = data['v']
            print('已载入训练完成的神经元')
            return

        # BP神经网络训练方法
        data = self.input_data.T
        reason = self.output_data.T
        # 初始化误差，迭代次数，输入数据
        e = 1
        times = 0
        data = np.vstack((data, np.ones(shape=data[0, :].shape)))
        # 迭代过程
        while e > 0.001:
            # 生成取样随机数
            num = np.arange(12)
            np.random.shuffle(num)
            for i in range(12):
                e_single = 0
                x = data[:, num[i]].reshape(15*15+1, 1)    # 抽取一个随机样本
                d = reason[:, num[i]]  # 随机样本对应的输出
                y = self.f(x.T.dot(self.v))
                # 将训练数据输入神经网络得到输出值
                y_work = np.hstack((y, np.array([1]).reshape(1, 1)))
                o = self.f(y_work.dot(self.w)).reshape(1, 12)
                # 残差平方和
                e_single += 1/2 * np.sum((d - o)**2)

                # 权值更新（误差反向传播）
                self.w += y_work.T.dot(self.k * (d - o) * o * (1 - o))
                self.v += x.dot(self.k * np.sum((d - o) * o * (1 - o) * self.w[0:500, :], axis=1) * y * (1 - y))
                # 记录迭代次数
                times += 1

            e = e_single

            # 每迭代一千次输出一次准确率
            if times % 1200 == 0 or e < 0.001:
                y = self.f(data.T.dot(self.v))
                # 将训练数据输入神经网络得到输出值
                y_work = np.hstack((y, np.ones([12, 1])))
                o = self.f(y_work.dot(self.w))
                # 将大于等于0.5的值映射为1，小于0.5的值映射为0
                o[o < 0.5] = 0
                o[o >= 0.5] = 1
                # 计算准确率
                p = np.sum((np.argwhere(o == 1) == np.argwhere(reason == 1)))/24.0
                print("正确率：", p)
                if e < 0.001:
                    print("训练完成，最终误差为：", e)
                else:
                    print("训练误差为：", e)

        so.savemat('Data.mat', {'w': self.w, 'v': self.v})

    def img_to_data(self, img):
        # 提取图片数据
        img_data = np.array(img).reshape(1, 15 * 15)
        img_data[img_data > 0] = 1
        return img_data

    def distinguish(self, img):
        # 识别方法

        data = np.hstack((self.img_to_data(img), np.array([1]).reshape([1, 1])))
        y = self.f(data.dot(self.v))
        # 将训练数据输入神经网络得到输出值
        y_work = np.hstack((y, np.array([1]).reshape(1, 1)))
        o = self.f(y_work.dot(self.w)).reshape(1, 12)
        # 将大于等于0.5的值映射为1，小于0.5的值映射为0
        o[o < 0.5] = 0
        o[o >= 0.5] = 1
        # 将结果转化为整型
        reason = np.sum(np.argwhere(o == 1)) + 1

        return reason


if __name__ == '__main__':
    BP = BPNetwork()
    BP.pretreatment()
    BP.train()
