import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.SegMatting.models.submodules import ConvLayer, UpsampleConvLayer, ResidualBlock, UpConvLayerShuffle, SegmentNet, CBAM, SpatialGate2, TransposedConvLayer

from Models.SegMatting.backbones import BACKBONE_LIST


#
# Basic UNet
#
def skip_sum(x1, x2):
    return x1 + x2

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

class BaseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self, append=None):
        if append is None:
            append = [0, 0, 0, 0]
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        index = 0
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size + append[index] if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2 + append[index+1],
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)


class UNet(BaseUNet):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNet, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                   num_encoders, base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size, kernel_size=5,
                                           stride=2, padding=2, norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img

class UNetMatting(BaseUNet):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=False):
        super(UNetMatting, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                   num_encoders, base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        backbone_features = [ 0, 16, 24, 0]

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W
        self.backbone = BACKBONE_LIST['mobilenetv2'](num_input_channels)
        self.backbone.load_pretrained_ckpt()
        self.segnet = SegmentNet(self.backbone)

        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        index = 0
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size+backbone_features[index], output_size, kernel_size=5,
                                           stride=2, padding=2, norm=self.norm))
            self.attentions.append(CBAM(output_size, 16))
            index += 1

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        segment, seg_8, seg_features = self.segnet(x)
        seg_2, seg_4 = seg_features
        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            if i ==1:
                x = torch.cat([x, seg_2], dim=1)
            if i == 2:
                x = torch.cat([x, seg_4], dim=1)

            x = encoder(x)
            """s_wise = self.attentions[i](x)
            x = x * s_wise"""
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return segment, img


class SegMatting(nn.Module):
    def __init__(self, segment_size=[16, 24, 32], pretrained=True):
        super(SegMatting, self).__init__()
        self.input_c = 3
        self.output_c = 1
        self.pretrained = pretrained
        self.base = 64
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
        )
        self.backbone = BACKBONE_LIST['mobilenetv2'](self.input_c)
        self.segmentNet = SegmentNet(self.backbone)
        self.base_feature = nn.Sequential(
            nn.Conv2d(segment_size[-1], self.base, 7, stride=1, padding=3),
            nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.02),
            nn.Conv2d(self.base, self.base, 7, stride=1, padding=3),
            nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.02)
        )
        self.spatial_gate = SpatialGate2(self.base)
        channel = self.base
        self.predict_block = nn.Sequential(
            nn.Conv2d(channel, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 1, 3, stride=1, padding=1)
        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)
        if self.pretrained:
            self.backbone.load_pretrained_ckpt()
    def forward(self, input):
        segment, seg_8, _ = self.segmentNet(input)
        input_feature = self.feature_extractor(input)
        seg_from8_2 = F.interpolate(seg_8, scale_factor=4, align_corners=False, mode="bilinear")
        seg_from8_2 = self.base_feature(seg_from8_2)
        mask = self.spatial_gate(seg_from8_2)
        input_feature = input_feature

        high_feature = input_feature

        alpha = self.predict_block(high_feature)
        alpha = torch.sigmoid(alpha)
        alpha = F.interpolate(alpha, scale_factor=2, align_corners=False, mode="bilinear")
        return segment, alpha

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
