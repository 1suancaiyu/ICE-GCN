import torch.nn as nn
import torch.nn.functional as F
import torch

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class Motion_Channel_Excitation(nn.Module):
    def __init__(self, in_channels, out_channels, n_segment=3):
        super(Motion_Channel_Excitation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_segment = n_segment
        self.reduced_channels = self.in_channels // 16
        self.pad = (0, 0, 0, 0, 0, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        # layers
        self.me_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.me_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.me_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1)
        self.me_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        print('=> Using Motion_Excitation')

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()

        # get n_batch
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        nt, c, v = x.size()
        n_batch = nt // self.n_segment

        # squeeze conv
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        x = self.me_squeeze(x)
        x = self.me_bn1(x)
        n, c_r, t, v = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c_r, v)

        # temporal split
        nt, c_r, v = x.size()
        x_plus0, _ = x.view(n_batch, self.n_segment, c_r, v).split([self.n_segment - 1, 1], dim=1)  # x(t) torch.Size([2000, 2, 4, 25])

        # x(t+1) conv
        x = x.view(n, t, c_r, v).permute(0, 2, 1, 3).contiguous()
        x_plus1 = self.me_conv1(x)
        x_plus1 = x_plus1.permute(0, 2, 1, 3).contiguous().view(n * t, c_r, v)
        _, x_plus1 = x_plus1.view(n_batch, self.n_segment, c_r, v).split([1, self.n_segment - 1], dim=1)  # x(t+1) torch.Size([2000, 2, 4, 25])

        # subtract
        x_me = x_plus1 - x_plus0  # torch.Size([2000, 2, 4, 25]) torch.Size([2000, 2, 4, 25])

        # pading
        x_me = F.pad(x_me, self.pad, mode="constant", value=0)  # torch.Size([2000, 2, 4, 25]) -> orch.Size([2001, 2, 4, 25])

        # spatical pooling
        x_me = x_me.view(n,t,c_r,v).permute(0, 2, 1, 3).contiguous()
        x_me = self.avg_pool(x_me)

        # expand
        x_me = self.me_expand(x_me)  # torch.Size([6000, 64, 1])

        # sigmoid
        x_me = x_me.permute(0, 2, 1, 3).contiguous().view(n * t, c, 1)
        x_me_score = self.sigmoid(x_me)

        x = x_origin * x_me_score # (nt,c,v) * (nt, c, 1)
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        return x

class Short_Term_Channel_Excitation(nn.Module):
    def __init__(self, in_channels, stride=1, sq_scale=16, n_segment=3):
        super(Short_Term_Channel_Excitation, self).__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.stride = stride

        self.reduced_channels = self.in_channels // sq_scale

        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.action_p2_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.action_p2_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)

        print(">>>>>>   Short_Term_Channel_Excitation")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        n_batch = n * t // self.n_segment
        x_origin = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)

        # 2D convolution: c*T*1*1, channel excitation
        x_ce = self.avg_pool(x) # spatial pooling
        x_ce = self.action_p2_squeeze(x_ce)

        _, c_r, _, _ = x_ce.size()
        x_ce = x_ce.view(n_batch, self.n_segment, c_r, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        x_ce = self.action_p2_conv1(x_ce)
        x_ce = self.relu(x_ce)

        # reshape
        x_ce = x_ce.transpose(2,1).contiguous().view(-1, c_r, 1)

        # expand
        x_ce = x_ce.view(n, t, c_r, 1).permute(0, 2, 1, 3).contiguous()
        x_ce = self.action_p2_expand(x_ce)
        x_ce = self.sigmoid(x_ce)
        x_ce = x_ce.permute(0, 2, 1, 3).contiguous().view(n * t, c, 1)

        # merge
        x_ce = x_origin * x_ce + x_origin # (nt,c,v) * (nt, c, 1)

        # out
        x_ce = x_ce.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()

        return x_ce

# Shift-GCN worked skeleton Temporal_Channel_Excitation
class Shiftgcn_Temporal_Channel_Excitation(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_segment=3):
        super(Shiftgcn_Temporal_Channel_Excitation, self).__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels == 3:
            self.reduced_channels = 3
        else:
            self.reduced_channels = self.in_channels // 16

        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.action_p2_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.action_p2_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)
        self.action_p2_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=(stride, 1))
        self.bn = nn.BatchNorm2d(self.out_channels)
        print(">>>>>>>>>>>> Shiftgcn_Temporal_Channel_Excitation")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        n_batch = n * t // self.n_segment
        x_origin = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)

        # 2D convolution: c*T*1*1, channel excitation
        x_ce = self.avg_pool(x)
        x_ce = self.action_p2_squeeze(x_ce)

        _, c_r, _, _ = x_ce.size()
        x_ce = x_ce.view(n_batch, self.n_segment, c_r, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        x_ce = self.action_p2_conv1(x_ce)
        x_ce = self.relu(x_ce)

        # reshape
        x_ce = x_ce.transpose(2,1).contiguous().view(-1, c_r, 1)

        # expand
        x_ce = x_ce.view(n, t, c_r, 1).permute(0, 2, 1, 3).contiguous()
        x_ce = self.action_p2_expand(x_ce)
        x_ce = self.sigmoid(x_ce)
        x_ce = x_ce.permute(0, 2, 1, 3).contiguous().view(n * t, c, 1)

        # merge
        x_ce = x_origin * x_ce + x_origin

        # out
        x_ce = x_ce.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()

        # recover channel
        x_ce = self.action_p2_out(x_ce)
        x_ce = self.bn(x_ce)

        return x_ce

class Long_Term_Channel_Excitation(nn.Module):
    def __init__(self, in_planes):
        super(Long_Term_Channel_Excitation, self).__init__()
        self.in_planes = in_planes

        self.pooling = nn.AdaptiveAvgPool2d((None, 1))

        self.long_term = nn.Sequential(
            nn.Conv1d(self.in_planes, self.in_planes // 16, kernel_size=1),
            nn.BatchNorm1d(self.in_planes // 16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.in_planes // 16, self.in_planes, kernel_size=1),
            nn.Sigmoid())

        print('=> Using Long_Term_Channel_Excitation')

    def forward(self, x):
        x_origin = x
        # x = self.pooling(x)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n*v, c, t)
        long_term_weight = self.long_term(x)  # n*v, c, t

        # mean all the frames get the longterm features and add to each frames
        long_term_weight = long_term_weight.mean(-1).unsqueeze(-1)

        x = x * long_term_weight

        # excitation
        x = x.view(n, v, c, t).permute(0, 2, 3, 1)  # n,t,c,v
        return x

# Spatial_Channel_Excitation
class Spatial_Channel_Excitation(nn.Module):
    def __init__(self):
        super(Spatial_Channel_Excitation, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>  Spatial_Channel_Excitation")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        x = x.mean(1, keepdim=True)
        x = self.conv(x)
        x_spatical_score = self.sigmoid(x)
        x = x_origin * x_spatical_score
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        return x

class multi_scale_channel_excitation(nn.Module):
    def __init__(self, out_channels):
        super(multi_scale_channel_excitation, self).__init__()
        self.mce = Motion_Channel_Excitation(out_channels, out_channels, n_segment=3)
        self.stce = Short_Term_Channel_Excitation(out_channels, n_segment=3)
        # self.sce = Spatical_Channel_Excitation(out_channels=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)

    def forward(self, x):
        x_sce = x + self.sce(x)
        x_mce = x + self.mce(x)
        x_stce = x + self.stce(x)

        # x = self.bn(x_stce + x_mce )
        x = self.bn( x_sce + x_mce + x_stce )
        return x

# Temporal_Channel_Excitation_conv1d_two_layer
class Temporal_Channel_Excitation_conv1d_two_layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_segment=4):
        super(Temporal_Channel_Excitation_conv1d_two_layer, self).__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduced_channels = self.in_channels // 16

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.tce_tmp_conv = nn.Conv1d(self.in_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.tce_expand = nn.Conv1d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        n_batch = n * t // self.n_segment
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)

        # spatial pooling
        x_tce = self.avg_pool(x)

        # reshape each group 3 frame
        x_tce = x_tce.view(n_batch, self.n_segment, c, 1).squeeze(-1).transpose(2, 1).contiguous()
        # temporal conv
        x_tce = self.tce_tmp_conv(x_tce)
        _,c_r,_ = x_tce.size()
        # relu as SEnet
        x_tce = self.relu(x_tce)

        # reshape
        x_tce = x_tce.transpose(2,1).contiguous().view(-1, c_r, 1)

        # 1D convolution, channel expand
        x_tce = self.tce_expand(x_tce)
        # get importance weight for channel dim
        x_tce = self.sigmoid(x_tce)

        # excite channel dim
        x_tce = x_origin * x_tce + x_origin

        # reshape as origin input
        x_tce = x_tce.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()

        x_tce = self.bn(x_tce)

        return x_tce