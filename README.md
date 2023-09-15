# 第三届计图挑战赛赛题一：风格及语义引导的风景图片生成 第九名方案 基于jittor实现的面向风格迁移的SEAN方法（改进版）

![主要结果](https://s3.bmp.ovh/imgs/2023/09/15/471f917e278cce75.jpg)

![主要框架](https://s3.bmp.ovh/imgs/2023/09/15/ef16fe25a9c88479.png)

## 简介
| 简单介绍项目背景、项目特点

本项目包含了第三届计图挑战赛赛题一 - 风格及语义引导的风景图片生成比赛的代码实现。本项目的特点是：采用了改进后的SEAN方法对10000张训练集图像进行训练，取得了良好的图像生成质量和风格迁移效果。

## 安装 
| 介绍基本的硬件需求、运行环境、依赖安装方法

本项目可在 1 张 3090 上运行，batchsize=3训练110个epoch的时间约96小时。也可使用mpi指令在多卡上加速训练。
 4090训练速度更快，但不支持多卡并行训练。
 以下为本项目在4090上训练时的性能测试：
 batchsize=1 占用显存12.5G  约 41min/epoch
 batchsize=2 占用显存16.4G  约 30min/epoch
 batchsize=3 占用显存20.3G  约 25min/epoch

#### 运行环境
- python >= 3.7
- opencv_python==4.8.0.76
- jittor==1.3.7.7

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型
预训练模型模型下载地址为 https://pan.baidu.com/s/1uuA4x24eMkr89P1gDUu-oA?pwd=3r4r ，下载后放入目录 `<root>/pretrained/`下， 
我们在链接中同时也提供了本次比赛提交的最佳模型，同样下载后放入目录`<root>/checkpoint/` 。

## 训练
｜ 介绍模型训练的方法

单卡训练可运行以下命令：
```
python train.py
```

多卡训练可以运行以下命令：
```
CUDA_VISIBLE_DEVICES="0,1" mpirun --allow-run-as-root -np 2 python3.7 -m train
```

## 推理
｜ 介绍模型推理、测试、或者评估的方法

生成测试集上的结果可以运行以下命令：

```
python test.py
```

## 致谢
| 对参考的论文、开源库予以致谢，可选

此项目基于论文 *SEAN: Image Synthesis With Semantic Region-Adaptive Normalization* 实现，部分代码参考了 [SEAN](https://github.com/zhang-zx/SEAN-PyTorch/tree/master)和 [Jittor](https://gitlink.org.cn/Ewtqf2i3v/surrenderbygugugu_pictures)。
