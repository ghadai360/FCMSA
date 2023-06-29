
from .attention import *



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class TopkPool(nn.Module):
    def __init__(self):
        super(TopkPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        topkv, _ = x.topk(5, dim=-1)
        return topkv.mean(dim=-1)
    
class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LeftMulScBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LeftMulScBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv1_2_3 = conv3x3(scale_width, scale_width)
        self.bn1_2_3 = norm_layer(scale_width)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)

        
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        ##########################################################
        '''Left side'''
        out_1_1 = self.conv1_2_1(sp_x[0])
        out_1_1 = self.bn1_2_1(out_1_1)
        out_1_1_relu = self.relu(out_1_1)
        out_1_2 = self.conv1_2_2(out_1_1_relu + sp_x[1])
        out_1_2 = self.bn1_2_2(out_1_2)
        out_1_2_relu = self.relu(out_1_2)
        out_1_3 = self.conv1_2_3(out_1_2_relu + sp_x[2])
        out_1_3 = self.bn1_2_3(out_1_3)
        out_1_3_relu = self.relu(out_1_3)
        out_1_4 = self.conv1_2_4(out_1_3_relu + sp_x[3])
        out_1_4 = self.bn1_2_4(out_1_4)
        out = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)
        

        out = self.relu(out)

        return out
    
    
class RightMulScBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(RightMulScBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv2_2_1 = conv3x3(scale_width, scale_width)
        self.bn2_2_1 = norm_layer(scale_width)
        self.conv2_2_2 = conv3x3(scale_width, scale_width)
        self.bn2_2_2 = norm_layer(scale_width)
        self.conv2_2_3 = conv3x3(scale_width, scale_width)
        self.bn2_2_3 = norm_layer(scale_width)
        self.conv2_2_4 = conv3x3(scale_width, scale_width)
        self.bn2_2_4 = norm_layer(scale_width)

        
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        ##########################################################
        '''Right side'''
        out_2_1 = self.conv2_2_1(sp_x[3])
        out_2_1 = self.bn2_2_1(out_2_1)
        out_2_1_relu = self.relu(out_2_1)
        out_2_2 = self.conv2_2_2(out_2_1_relu + sp_x[2])
        out_2_2 = self.bn2_2_2(out_2_2)
        out_2_2_relu = self.relu(out_2_2)
        out_2_3 = self.conv2_2_3(out_2_2_relu + sp_x[1])
        out_2_3 = self.bn2_2_3(out_2_3)
        out_2_3_relu = self.relu(out_2_3)
        out_2_4 = self.conv2_2_4(out_2_3_relu + sp_x[0])
        out_2_4 = self.bn2_2_4(out_2_4)
        out = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=1)

        out = self.relu(out)

        return out
class FDM(nn.Module):
    def __init__(self):
        super(FDM, self).__init__()
        self.factor = round(1.0/(28*28), 3)

    def forward(self, fm1, fm2):
        b, c, w1, h1 = fm1.shape
        _, _, w2, h2 = fm2.shape
        fm1 = fm1.view(b, c, -1) # B*C*S
        fm2 = fm2.view(b, c, -1) # B*C*M

        fm1_t = fm1.permute(0, 2, 1) # B*S*C

        # may not need to normalize
        fm1_t_norm = F.normalize(fm1_t, dim=-1)
        fm2_norm = F.normalize(fm2, dim=1)
        M = -1 * torch.bmm(fm1_t_norm, fm2_norm) # B*S*M

        M_1 = F.softmax(M, dim=1)
        M_2 = F.softmax(M.permute(0, 2, 1), dim=1)
        new_fm2 = torch.bmm(fm1, M_1).view(b, c, w2, h2)
        new_fm1 = torch.bmm(fm2, M_2).view(b, c, w1, h1)

        return self.factor*new_fm1,self.factor* new_fm2
    
class FCMSA(nn.Module):

    def __init__(self, block_b, block_L,block_R, layers, num_classes=7): #num_classes=12666
        super(MPMA, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0])
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)


        # In this branch, each BasicBlock replaced by MulScaleBlock.
        self.layer3_1 = self._make_layer(block_L, 128, 256, layers[2], stride=2)
        self.layer3_2 = self._make_layer(block_R, 128, 256, layers[2], stride=2)
        #self.layer3_3 = self._make_layer(block_L, 128, 256, layers[2], stride=2)
    
        
        self.layer4_1 = self._make_layer(block_L, 256, 512, layers[3], stride=2)
        self.layer4_2 = self._make_layer(block_R, 256, 512, layers[3], stride=2)

        self.eSEModule3_1 =eSEModule(256, 4)
        self.SpatialGate4_1 = SpatialGate()
        #self.eSEModule3_2 =eSEModule(256, 4)
        self.eSEModule3_2 =eSEModule(256, 4)
        self.SpatialGate4_2 = SpatialGate()

        
        self.inter = FDM()
        
        self.CGFusion= ChannelGate(1024, 32) #32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3_1 = nn.Linear(1024, 512)
        self.fc3_2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        
        
        branch_1_out = self.layer3_1(out)
        branch_2_out = self.layer3_2(out)
        branch_1_out=self.eSEModule3_1(branch_1_out)
        branch_2_out=self.eSEModule3_2(branch_2_out)
        
        branch_1_out = self.layer4_1(branch_1_out)
        branch_2_out = self.layer4_2(branch_2_out)

        #Attention
        branch_1_out=self.SpatialGate4_1(branch_1_out)
        branch_2_out=self.SpatialGate4_2(branch_2_out)

        
        new_d1_from2, new_d2_from1 = self.inter(branch_1_out, branch_2_out)
        gamma =[0.1,0.2,0.3,0.4,2.5,0.6,0.7,0.8,0.9,0.1]#0.5 #change hyperparameter here
        branch_1_out = branch_1_out + gamma[3]*(new_d1_from2)
        branch_2_out = branch_2_out + gamma[3]*(new_d2_from1)
        
        branch_3_out=torch.cat([branch_1_out, branch_2_out], dim=1) #feature level fusion
        branch_3_out=self.CGFusion(branch_3_out)
        

        branch_1_out = self.avgpool(branch_1_out)
        branch_1_out = torch.flatten(branch_1_out, 1)
        branch_2_out = self.avgpool(branch_2_out)
        branch_2_out = torch.flatten(branch_2_out, 1)
        
        #branch_3_out = torch.cat((branch_1_out, branch_2_out), -1) #Feture level fusion
        
        branch_3_out = self.avgpool(branch_3_out)
        branch_3_out = torch.flatten(branch_3_out, 1)
        
        x1=self.fc1(branch_1_out)
        x2=self.fc2(branch_2_out)
        x_concat=self.fc3_1(branch_3_out)
        x_concat=self.fc3_2(x_concat)
               

        return x1 , x2, x_concat

    def forward(self, x):
        return self._forward_impl(x)


def fcmsa():
    return FCMSA(block_b=BasicBlock, block_L=LeftMulScBlock,block_R=RightMulScBlock,layers=[2, 2, 1, 1])
