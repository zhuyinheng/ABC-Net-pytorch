import torch.nn as nn
import torch


class ABCConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0.0,
                 linear=False, base_number=3):
        super(ABCConv2d, self).__init__()
        assert base_number == 3 or base_number == 1, "support base_number == 3 or base_number == 1 "
        self.layer_type = 'ABC_Conv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.base_number = base_number
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.linear = linear
        if not self.linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            if self.base_number == 1:
                self.bases_conv2d_1 = nn.Conv2d(input_channels, output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            else:

                self.bases_conv2d_1 = nn.Conv2d(input_channels, output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
                self.bases_conv2d_2 = nn.Conv2d(input_channels, output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
                self.bases_conv2d_3 = nn.Conv2d(input_channels, output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            if self.base_number == 1:
                self.bases_linear_1 = nn.Linear(input_channels, output_channels)
            else:
                self.bases_linear_1 = nn.Linear(input_channels, output_channels)
                self.bases_linear_2 = nn.Linear(input_channels, output_channels)
                self.bases_linear_3 = nn.Linear(input_channels, output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        # x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if self.base_number == 1:
            if not self.linear:
                x = self.bases_conv2d_1(x)
            else:
                x = self.bases_linear_1(x)
        else:
            if not self.linear:
                x = self.bases_conv2d_1(x) + self.bases_conv2d_2(x) + self.bases_conv2d_3(x)
            else:
                x = self.bases_linear_1(x) + self.bases_linear_2(x) + self.bases_linear_3(x)
        x = self.relu(x)
        return x


BinConv2d = ABCConv2d


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, base_number=3):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.base_number = base_number
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(64, 192, kernel_size=5, stride=1, padding=2, groups=1, base_number=self.base_number),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(192, 384, kernel_size=3, stride=1, padding=1, base_number=self.base_number),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1, base_number=self.base_number),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1, base_number=self.base_number),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, linear=True, base_number=self.base_number),
            BinConv2d(4096, 4096, dropout=0.1, linear=True, base_number=self.base_number),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def my_model_loader(self, state_dict, strict=True):
        own_state = self.state_dict()
        # map fp model to ABC-Net
        load_map = \
            {
                'features.0.weight': 'features.0.weight',
                'features.0.bias': 'features.0.bias',
                'features.4.bases_conv2d_1.weight': 'features.3.weight',
                'features.4.bases_conv2d_1.bias': 'features.3.bias',
                'features.6.bases_conv2d_1.weight': 'features.6.weight',
                'features.6.bases_conv2d_1.bias': 'features.6.bias',
                'features.7.bases_conv2d_1.weight': 'features.8.weight',
                'features.7.bases_conv2d_1.bias': 'features.8.bias',
                'features.8.bases_conv2d_1.weight': 'features.10.weight',
                'features.8.bases_conv2d_1.bias': 'features.10.bias',
                'classifier.0.bases_linear_1.weight': 'classifier.1.weight',
                'classifier.0.bases_linear_1.bias': 'classifier.1.bias',
                'classifier.1.bases_linear_1.weight': 'classifier.4.weight',
                'classifier.1.bases_linear_1.bias': 'classifier.4.bias',
                'classifier.4.weight': 'classifier.6.weight',
                'classifier.4.bias': 'classifier.6.bias',
            }

        for k, v in load_map.items():
            own_state[k].copy_(state_dict[v].data)


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet_fp_pretrained.pth'
        pretrained_model = torch.load(model_path)
        model.my_model_loader(pretrained_model)
    return model
