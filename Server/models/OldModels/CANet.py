import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import Server.models.OldModels.AttModule as attnet

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class AddCoordinates(object):
    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        image = torch.cat((coords.to(image.device), image), dim=1)

        return image

class ConvBatchBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, dilation=1,  relu=True, bn=True, bias=False):
        super(ConvBatchBlock, self).__init__()
        self.out_c = out_c
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride,padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_c, skip_c, base=128):
        super(DeconvBlock, self).__init__()
        out_c = skip_c
        if out_c<base:#base 보다 작으면 base로, 아니면 기존의 채널로
            out_c = base
        self.upsampler =  nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = ConvBatchBlock(in_c + skip_c, out_c, 1, stride=1, padding=0, bias=False, relu=True)

    def forward(self, x, skip_con=None):
        x = self.upsampler(x)
        if skip_con is not None:
            x = torch.cat([x, skip_con], dim=1)
        x = self.conv1(x)
        return x

class WeakDeconv(nn.Module):
    def __init__(self, in_c, skip_c, base=128):
        super(WeakDeconv, self).__init__()
        out_c = skip_c
        if base<out_c:
            out_c = base
        self.upsampler = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = ConvBatchBlock(in_c + skip_c, out_c, 1, stride=1, padding=0, bias=False, relu=True)

    def forward(self, x, skip_con=None):
        x = self.upsampler(x)
        if skip_con is not None:
            x = torch.cat([x,skip_con], dim=1)
        x = self.conv1(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_c, us_c=128, bias=False, up_output=True, out_result=True):
        super(FPN, self).__init__()
        self.base = us_c
        self.up_output = up_output
        self.out_reuslt= out_result
        self.conv = ConvBatchBlock(in_c, self.base, 1, stride=1, padding=0, bias=bias)
        self.conv2 = ConvBatchBlock(self.base, self.base, 3, stride=1, padding=1, bias=bias)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.lastConv = nn.Conv2d(self.base, 1,1, stride=1, padding=0)
        self.bilinear = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    def forward(self, x, us):
        if x.shape[1]!=self.base:
            x = self.conv(x)
        x = x+us
        add = x = self.conv2(x)
        if self.up_output:
            us = self.upsampler(add)
        if self.out_reuslt:
            out = torch.sigmoid(self.lastConv(x))
            out = self.bilinear(out)
        else:
            out = None
        return add, us, out

#use DenseNet Encoder
class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model
        self.standard = 256
        self.module_list = nn.ModuleList([])
        for name, data in self.model.features.named_children():
            if name=="transition3":
                break
            self.module_list.append(data)
    def forward(self, x):
        l_channel = None
        for m in self.module_list:
            x = m(x)
            if x.shape[1] == self.standard and isinstance(m, models.densenet._DenseBlock):
                l_channel = x
        return x, l_channel

class DenseDecoder(nn.Module):
    def __init__(self):
        super(DenseDecoder, self).__init__()
        base = 128
        layers = [1920, 1792, 512, 256, 64]
        self.deconv1 = DeconvBlock(layers[0], layers[1])
        self.deconv2 = DeconvBlock(layers[1], layers[2])
        self.deconv3 = DeconvBlock(layers[2], layers[3])
        self.deconv4 = DeconvBlock(layers[3], layers[4])
        self.deconv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBatchBlock(base, base, 1, stride=1, padding=0, relu=True)
        )
        self.conv = ConvBatchBlock(layers[1], base, 1, stride=1, padding=0, bias=True, relu=True)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        self.fpn1 = FPN(layers[2])
        self.fpn2 = FPN(layers[3])
        self.fpn3 = FPN(layers[4])
        self.fpn4 = FPN(layers[4], up_output=False)


    def forward(self, x, skip_conn):
        R1, R2, R3, R4 = skip_conn

        x4 = self.deconv1(x, R4)
        x3 = self.deconv2(x4, R3)
        x2 = self.deconv3(x3, R2)
        x1 = self.deconv4(x2, R1)
        x = self.deconv0(x1)
        us4 = self.conv(x4)
        us4 = self.upsampler(us4)
        _, us3, out8 = self.fpn1(x3, us4)
        _, us2, out4 = self.fpn2(x2, us3)
        _, us1, out2 = self.fpn3(x1, us2)
        add, _, out = self.fpn4(x, us1)
        return [out, out2, out4, out8, add]


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        base = 128
        layers = [1920, 1792, 512, 256, 64]
        self.deconv1 = DeconvBlock(layers[0], layers[1])
        self.deconv2 = DeconvBlock(base, layers[2])
        self.deconv3 = DeconvBlock(base, layers[3])
        self.deconv4 = DeconvBlock(base, layers[4])
        #deconv 4's output is 64 x 256 x 256
        self.deconv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBatchBlock(base, base, 1, stride=1, padding=0, relu=True)
        )
        self.conv = ConvBatchBlock(layers[1], base, 1, stride=1, padding=0, bias=True, relu=True)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        self.fpn1 = FPN(layers[2])
        self.fpn2 = FPN(layers[3])
        self.fpn3 = FPN(layers[4])
        self.fpn4 = FPN(layers[4], up_output=False)


    def forward(self, x, skip_conn):
        R1, R2, R3, R4 = skip_conn

        x4 = self.deconv1(x, R4)
        x3 = self.deconv2(x4, R3)
        x2 = self.deconv3(x3, R2)
        x1 = self.deconv4(x2, R1)
        x = self.deconv0(x1)
        us4 = self.conv(x4)
        us4 = self.upsampler(us4)
        _, us3, out8 = self.fpn1(x3, us4)
        _, us2, out4 = self.fpn2(x2, us3)
        _, us1, out2 = self.fpn3(x1, us2)
        add, _, out = self.fpn4(x, us1)
        return [out, out2, out4, out8, add]


class LABlock(nn.Module):
    def __init__(self, in_c, basic_c=128):
        super(LABlock, self).__init__()

    def forward(self,x):
        return x

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.convRGB = ConvBatchBlock(3, 256, 3, stride=1, padding=1)
        filters = [256+2, 256, 128, 128, 64, 256]
        layers=nn.ModuleList([])

        for idx in range(len(filters)-1):
            conv = ConvBatchBlock(filters[idx], filters[idx+1], 3, stride=1, padding=1, relu=True)
            layers.append(conv)
        self.fusion = nn.Sequential(*layers)
        self.last_conv = nn.Conv2d(filters[-1], 1, 1,  stride=1, padding=0)

    def forward(self, fg, bg, input):
        in_feature = self.convRGB(input)
        feature = torch.cat([fg, bg, in_feature], dim=1)
        print(feature.shape)
        x = self.fusion(feature)
        x = torch.sigmoid(self.last_conv(x))
        return x


class SegmentNet(nn.Module):
    def __init__(self, model):
        super(SegmentNet, self).__init__()
        self.encoder = Encoder(model)
        self.fg_decoder = DenseDecoder()
        self.bg_decoder = DenseDecoder()


    def forward(self, img):
        feature, scon = self.encoder(img)
        fg = self.fg_decoder(feature, scon)
        bg = self.bg_decoder(feature, scon)
        return fg, bg

class CANet(nn.Module):
    def __init__(self, model):
        super(CANet, self).__init__()
        basic = 256
        mid = 256
        high = 1792
        self.encoder = Encoder(model)
        self.dpa = attnet.DPA(high, mid)
        self.attConv = ConvBatchBlock(basic, basic, 3, stride=1, padding=1, relu=True)
        self.mixNet = nn.Conv2d(basic+mid, 1, 3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=4,mode='bilinear', align_corners=True)
    def forward(self, x):
        h_feature, l_feature = self.encoder(x)
        s_att, c_att = self.dpa(h_feature)
        w_feature = s_att*l_feature
        x = self.attConv(w_feature)
        x = torch.cat([x, c_att], dim=1)
        alpha = torch.sigmoid(self.mixNet(x))
        alpha = self.upsample2(alpha)
        w_feature = torch.mean(s_att, dim=1).unsqueeze(1)
        w_feature = self.upsample2(w_feature)
        return w_feature


class Discriminator(nn.Module):
    def __init__(self, input_c):
        super(Discriminator, self).__init__()
        #input shape is 512 512
        channels = [64, 128, 256, 256, 512, 1]
        layer = nn.ModuleList([])
        current_c = input_c
        for c in channels:
            layer.append(
                nn.Sequential(
                    ConvBatchBlock(current_c, c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(c),
                    nn.LeakyReLU(0.01)
                )
            )
            current_c = c
        self.model = nn.Sequential(*layer)
    def forward(self, x):
        return torch.sigmoid(self.model(x))