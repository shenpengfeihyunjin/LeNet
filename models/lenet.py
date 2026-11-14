import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 卷积神经网络
    用于 MNIST 手写数字识别

    输入尺寸：1×32×32
    输出尺寸：10
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # C1：第一层卷积，1 -> 6
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1
        )

        # S2：平均池化
        self.pool1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # C3：第二层卷积，6 -> 16
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1
        )

        # S4：平均池化
        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # C5：第三层卷积，16 -> 120
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1
        )

        # F6：全连接层
        self.fc1 = nn.Linear(120, 84)

        # 输出层：10个数字类别
        self.fc2 = nn.Linear(84, num_classes)

        self.activation = nn.Tanh()

    def forward(self, x):
        # 输入：1×32×32
        x = self.activation(self.conv1(x))
        x = self.pool1(x)

        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        x = self.activation(self.conv3(x))

        # 展平
        x = torch.flatten(x, start_dim=1)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = LeNet5()
    test_input = torch.randn(2, 1, 32, 32)
    output = model(test_input)

    print(model)
    print("输入尺寸：", test_input.shape)
    print("输出尺寸：", output.shape)
    print("模型参数量：", sum(p.numel() for p in model.parameters()))
