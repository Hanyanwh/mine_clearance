# mine_clearanceds
<h2>AI扫雷程序</h2>
<hr />
<ul>
    <li>现在的版本可以完成任意规模的扫雷地图</li>
    <li>扫描地图原理的采用训练好的BP神经网络将地图解释为两个矩阵，一个权值矩阵，一个数据矩阵</li>
    <li>由于分辨率的因素resouce中的图片是15*15的图片，在不同的机器上其图片大小可能不同，需要重新训练模型</li>
    <li>sampling.py是一个测试脚本，里面是零零散散的测试语句，主要是测试截图识别是否准确，如果不准确需要调整参数和重新训练模型</li>
    <li>Data.mat是已经训练好的模型</li>
    <li>Game.py是主程序在参数模型没有问题的情况下直接运行</li>
    <li>Neural_network.py是2层的BP神经网络模型/li>
    <li>Good luck</li>
</ul>