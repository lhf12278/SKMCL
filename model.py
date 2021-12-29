import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x1 = self.base.layer2(x)
        x = self.base.layer3(x1)
        x = self.base.layer4(x)
        return x, x1

class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_dim, in_dim // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_dim // 16, in_dim, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out

class Cross_modal_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, class_num, load_batch):
        super(Cross_modal_Attn, self).__init__()
        self.chanel_in = in_dim
        self.loader_batch = load_batch

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query1_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key1_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value1_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.bottleneck = nn.BatchNorm1d(self.chanel_in)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(self.chanel_in, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_cat):
        """
        inputs :
            x : input feature maps (B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        x = x_cat[:self.loader_batch]
        x1 = x_cat[self.loader_batch:]
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        proj_query1 = self.query1_conv(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key1 = self.key1_conv(x1).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        proj_value1 = self.value1_conv(x1).view(m_batchsize, -1, width * height)  # B X C X N

        relation = torch.bmm(proj_query, proj_key1)
        CM_attention = self.softmax(relation)

        x_out = torch.bmm(proj_value, CM_attention.permute(0, 2, 1))
        x_out = x_out.view(m_batchsize, C, width, height)
        x1_out = torch.bmm(proj_value1, CM_attention.permute(0, 2, 1))
        x1_out = x1_out.view(m_batchsize, C, width, height)

        out = torch.cat((x_out, x1_out), dim=0)

        out = out + x_cat
        out_pool = self.avgpool(out)
        out_pool = out_pool.view(out_pool.size(0), out_pool.size(1))
        out_feat = self.bottleneck(out_pool)

        if self.training:
            return out, self.classifier(out_feat), out_pool
        else:
            return out, self.classifier(out_feat), out_pool

class embed_net(nn.Module):
    def __init__(self,  low_dim,  class_num, drop=0.2, part = 3, alpha=0.2, nheads=4, arch='resnet50', wpa = False, load_batch=0):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.part = part
        self.lpa = wpa
        self.load_batch =load_batch

        self.cm_att = Cross_modal_Attn(in_dim=2048, class_num=class_num, load_batch=load_batch)
        self.sa = Self_Attn(in_dim=2048)

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        # domain specific block
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared four blocks
        x, x_low = self.base_resnet(x)

        x_map = x
        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat  = self.bottleneck(x_pool)

        if self.training:
            x_middle = x_low
            vt_mix = (x_middle[:self.load_batch] + x_middle[self.load_batch:])/2
            vt_mix1 = self.base_resnet.base.layer3(vt_mix)
            vt_mix2 = self.base_resnet.base.layer4(vt_mix1)

            vt_mix_pool = self.avgpool(vt_mix2)
            vt_mix_pool = vt_mix_pool.view(vt_mix_pool.size(0), vt_mix_pool.size(1))
            vt_mix_feat = self.bottleneck(vt_mix_pool)

            return x_pool, self.classifier(feat), x, self.classifier(vt_mix_feat), vt_mix_pool
        else:
            return self.l2norm(feat)