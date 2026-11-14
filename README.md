# LeNet-5：MNIST 手写数字识别实验

## 1. 项目简介

本项目基于 PyTorch 实现 LeNet-5 卷积神经网络，以 MNIST 手写数字数据集为研究对象，完成模型训练、测试集评估和预测结果可视化。

实验使用 GPU 进行训练，记录训练过程中的损失和准确率，并保存最佳模型权重、训练日志、分类报告、混淆矩阵及预测示例。

本项目采用经典 LeNet-5 的核心结构进行简化实现，使用普通卷积层实现 C3 层，并非对原始 LeNet-5 部分连接结构的严格复刻。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 20.04.6 |
| Python | 3.13 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GPU |
| 深度学习框架 | PyTorch |
| 数据集 | MNIST |

## 3. 网络结构

模型输入为单通道的 32×32 图像，采用 Tanh 激活函数和平均池化层。

网络主要结构如下：

1. 输入层：1×32×32
2. C1：5×5 卷积，1 通道映射至 6 通道
3. S2：2×2 平均池化
4. C3：5×5 卷积，6 通道映射至 16 通道
5. S4：2×2 平均池化
6. C5：5×5 卷积，16 通道映射至 120 通道
7. F6：全连接层，120 映射至 84
8. 输出层：全连接层，84 映射至 10 个数字类别

模型参数量：61,706。

## 4. 数据集

实验使用 MNIST 手写数字数据集：

- 训练集：60,000 张图像
- 测试集：10,000 张图像
- 类别：0～9，共 10 类
- 原始图像尺寸：28×28
- 输入图像尺寸：32×32

预处理包括：

- 将图像转换为 Tensor；
- 使用 2 像素填充，将 28×28 图像扩展至 32×32；
- 使用 MNIST 均值 0.1307 和标准差 0.3081 进行标准化。

MNIST 原始数据未包含在 GitHub 仓库中。运行前需要将标准 MNIST 数据文件放置于 `datasets/MNIST/raw/` 目录。

## 5. 训练配置

| 参数 | 设置 |
|---|---|
| 训练轮数 | 20 |
| Batch Size | 128 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 损失函数 | CrossEntropyLoss |
| 随机种子 | 42 |
| 运行设备 | CUDA:1 |

训练过程中记录每轮训练损失、测试损失和测试准确率，并保存最佳模型与最后一轮模型。

## 6. 实验结果

最佳模型出现在第 16 轮。

| 评价指标 | 实验结果 |
|---|---:|
| 最佳轮次 | 16 |
| 测试集数量 | 10,000 |
| 测试损失 | 0.038253 |
| Accuracy | 99.03% |
| Macro Precision | 99.03% |
| Macro Recall | 99.02% |
| Macro F1-score | 99.03% |
| Weighted F1-score | 99.03% |

各类别均取得较高的分类指标，模型具有较好的 MNIST 手写数字识别能力。

### 结果可视化

训练和评估生成以下图片：

- `outputs/plots/lenet_loss_curve.png`：训练与测试损失曲线；
- `outputs/plots/lenet_accuracy_curve.png`：测试准确率变化曲线；
- `outputs/plots/lenet_confusion_matrix.png`：测试集混淆矩阵；
- `outputs/predictions/lenet_prediction_examples.png`：40 张手写数字预测示例。

## 7. 项目目录

```text
LeNet/
├── models/
│   └── lenet.py
├── outputs/
│   ├── checkpoints/
│   │   ├── best_lenet.pth
│   │   └── last_lenet.pth
│   ├── plots/
│   │   ├── lenet_loss_curve.png
│   │   ├── lenet_accuracy_curve.png
│   │   └── lenet_confusion_matrix.png
│   ├── predictions/
│   │   └── lenet_prediction_examples.png
│   ├── confusion_matrix.csv
│   ├── test_results.txt
│   ├── train.log
│   └── training_log.csv
├── dataset.py
├── train.py
├── evaluate.py
├── .gitignore
└── README.md
```

## 8. 运行方法

安装依赖：

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

准备 MNIST 数据集后，训练模型：

```bash
python train.py
```

评估最佳模型：

```bash
python evaluate.py
```

训练和评估结果保存在 `outputs/` 目录中。

## 9. 模型文件说明

- `best_lenet.pth`：训练过程中保存的最佳模型权重；
- `last_lenet.pth`：最后一轮训练结束时保存的模型权重；
- `training_log.csv`：逐轮训练指标；
- `test_results.txt`：测试集分类评估报告；
- `confusion_matrix.csv`：混淆矩阵数据。

## 10. 实验说明与局限性

本实验使用官方 MNIST 测试集准确率选择最佳训练轮次，因此测试集参与了模型选择。最终报告中的 99.03% 是当前最佳模型在该测试集上的评估结果，不能视为完全独立的最终测试结果。

此外，本项目使用普通卷积实现 C3 层，与原始 LeNet-5 的部分连接设计存在差异。

## 11. 实验总结

本项目完成了 LeNet-5 核心结构的简化实现，并在 MNIST 数据集上完成训练和评估。模型取得 99.03% 的测试准确率，完成了模型权重保存、训练过程记录、分类指标统计以及结果可视化，为理解经典卷积神经网络的结构和训练流程提供了实验基础。
