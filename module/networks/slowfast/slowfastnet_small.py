import time
import torch
import torch.nn as nn

__all__ = ['slowfast_resnet10', 'slowfast_resnet18', 'slowfast_resnet34']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None, head_conv=1):
        super(BasicBlock, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block=BasicBlock, layers=(3, 4, 6, 3), class_num=10, dropout=0.5):
        super(SlowFast, self).__init__()

        self.fast_in_planes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=2, head_conv=3)

        self.lateral_p1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                      padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(16, 16 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                      padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                      padding=(2, 0, 0))

        self.slow_in_planes = 64 + 64 // 8 * 2
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_in_planes + 512, class_num, bias=False)

    def forward(self, x):
        fast, lateral = self.fast_path(x[:, :, ::2, :, :])
        slow = self.slow_path(x[:, :, ::16, :, :], lateral)
        x = torch.cat([slow, fast], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def slow_path(self, x, lateral):
        x = self.slow_conv1(x)  # 64c
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_max_pool(x)  # 64c
        x = torch.cat([x, lateral[0]], dim=1)  # 64+16c
        x = self.slow_res2(x)  # 64c
        x = torch.cat([x, lateral[1]], dim=1)  # 64+16c
        x = self.slow_res3(x)  # 128c
        x = torch.cat([x, lateral[2]], dim=1)  # 128+32c in-planes
        x = self.slow_res4(x)  # 256c
        x = torch.cat([x, lateral[3]], dim=1)  # 256c+64c
        x = self.slow_res5(x)  # 512c
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def fast_path(self, x):
        lateral = []
        x = self.fast_conv1(x)  # 8c
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_max_pool(x)
        lateral_p = self.lateral_p1(pool1)  # 16c
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)  # 8c
        lateral_res2 = self.lateral_res2(res2)  # 16c
        lateral.append(lateral_res2)

        res3 = self.fast_res3(res2)  # 16c
        lateral_res3 = self.lateral_res3(res3)  # 32c
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)  # 32c
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        down_sample = None
        if stride != 1 or self.fast_in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv3d(
                    self.fast_in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = list()
        layers.append(block(self.fast_in_planes, planes, stride, down_sample, head_conv=head_conv))
        self.fast_in_planes = planes * block.expansion
        for idx in range(1, blocks):
            layers.append(block(self.fast_in_planes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        down_sample = None
        if stride != 1 or self.slow_in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv3d(
                    self.slow_in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = list()
        layers.append(block(self.slow_in_planes, planes, stride, down_sample, head_conv=head_conv))
        self.slow_in_planes = planes * block.expansion
        for idx in range(1, blocks):
            layers.append(block(self.slow_in_planes, planes, head_conv=head_conv))

        self.slow_in_planes = planes * block.expansion + planes * block.expansion // 8 * 2
        return nn.Sequential(*layers)


def slowfast_resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    network = SlowFast(BasicBlock, [1, 1, 1, 1], **kwargs)
    return network


def slowfast_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    network = SlowFast(BasicBlock, [2, 2, 2, 2], **kwargs)
    return network


def slowfast_resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    network = SlowFast(BasicBlock, [3, 4, 6, 3], **kwargs)
    return network


def slowfast_resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def slowfast_resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(BasicBlock, [3, 4, 23, 3], **kwargs)
    return model


def slowfast_resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(BasicBlock, [3, 8, 36, 3], **kwargs)
    return model


def time_sync(device):
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.time()


def slowfast_test():
    num_classes = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is', device)
    input_tensor = torch.rand(1, 3, 32, 320, 320).to(device)
    model = slowfast_resnet34(class_num=num_classes)
    model.to(device)
    model.eval()
    start = time_sync(device)
    for i in range(100):
        output = model(input_tensor)
        print(output.size())
    end = time_sync(device)
    print((end - start) / 100)


if __name__ == "__main__":
    slowfast_test()
