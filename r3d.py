import torch
import torch.nn as nn

def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, planes, stride=1):
    return nn.Conv3d(in_planes,
                     planes, 
                     kernel_size = (3, 3, 3),
                     stride = stride, 
                     padding = 1,
                     bias = False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, 
                     out_planes, 
                     kernel_size = 1,
                     stride = stride, 
                     bias = False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, 
                 block, 
                 layers, 
                 block_inplanes, 
                 n_input_channels=3, 
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 n_classes=5,
                 dropout_probability=0.5):
        super().__init__()

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, 
                               self.in_planes,
                               kernel_size = (conv1_t_size, 7, 7),
                               stride = (conv1_t_stride, 2, 2),
                               padding = (conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(block_inplanes[3], 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        self.dropout = nn.Dropout(dropout_probability)
        # self.fc3 = nn.Linear(block_inplanes[3], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes, stride),
                nn.BatchNorm3d(planes))
        
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # x = self.fc3(x)

        return x
    

def generate_model(model_depth, **kwargs):
    assert model_depth in [10,18,34]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)

    return model

