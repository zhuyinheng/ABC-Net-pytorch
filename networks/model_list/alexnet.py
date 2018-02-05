import torch.nn as nn
import torch

class ABCConv2d(nn.Module):
    def __init__(self,input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False,base_number=2):
        super(ABCConv2d, self).__init__()
        assert base_number == 2 or base_number == 1, "support 2,1 only"
        self.layer_type = 'ABCConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.base_number=base_number
        self.bases=[]
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            if self.base_number==1:
                self.bases_conv2d_1 = nn.Conv2d(input_channels, output_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            else:

                self.bases_conv2d_1=nn.Conv2d(input_channels, output_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
                self.bases_conv2d_2 = nn.Conv2d(input_channels, output_channels,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            if self.base_number == 1:
                self.bases_linear_1 = nn.Linear(input_channels, output_channels)
            else:
                self.bases_linear_1=nn.Linear(input_channels, output_channels)
                self.bases_linear_2=nn.Linear(input_channels, output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        # x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if self.base_number==1:
            if not self.Linear:
                x=self.bases_conv2d_1(x)
            else:
                x = self.bases_linear_1(x)
        else:
            if not self.Linear:
                x=self.bases_conv2d_1(x)+self.bases_conv2d_2(x)
            else:
                x = self.bases_linear_1(x) + self.bases_linear_2(x)
        x = self.relu(x)
        return x
BinConv2d=ABCConv2d
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000,base_number=2):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.base_number=base_number
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1,base_number=self.base_number),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(256, 384, kernel_size=3, stride=1, padding=1,base_number=self.base_number),
            BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1,base_number=self.base_number),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1,base_number=self.base_number),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, Linear=True,base_number=self.base_number),
            BinConv2d(4096, 4096, dropout=0.1, Linear=True,base_number=self.base_number),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model
