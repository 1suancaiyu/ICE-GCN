import torch.nn as nn
import torch.nn.functional as F
import torch

class Motion_Excitation(nn.Module):
    def __init__(self, in_channels, n_segment=3):
        super(Motion_Excitation, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment

        self.reduced_channels = self.in_channels // 16

        self.pad = (0, 0, 0, 0, 0, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        # layers
        self.me_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.me_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.me_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1)
        self.me_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        print('=> Using Motion_Excitation')

    def forward(self, x):
        # get origin
        x_origin = x
        n, c, t, v = x.size()

        # get n_batch
        x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
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
        x_me = x_me.view(n, t, c_r, v).permute(0, 2, 1, 3).contiguous()
        x_me = self.avg_pool(x_me)

        # expand
        x_me = self.me_expand(x_me)  # torch.Size([6000, 64, 1])

        # sigmoid
        me_weight = self.sigmoid(x_me)
        x = x_origin * me_weight # n,c,t,v * n,c,t,1
        return x

class Long_Term_Excitation(nn.Module):
    def __init__(self, in_planes, joint_v=1):
        super(Long_Term_Excitation, self).__init__()
        self.in_planes = in_planes * joint_v

        self.pooling = nn.AdaptiveAvgPool2d((None, 1))

        self.long_term = nn.Sequential(
            nn.Conv1d(self.in_planes, self.in_planes // 16, kernel_size=1),
            nn.BatchNorm1d(self.in_planes // 16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.in_planes // 16, self.in_planes, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        x_origin = x
        x = self.pooling(x)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        long_term_weight = self.long_term(x).view(n, v, c, t).permute(0, 2, 3, 1)  # n,v,t,c  # n, v*c, t
        x = x_origin * long_term_weight
        return x

class Short_Term_Excitation(nn.Module):
    def __init__(self, in_channels):
        super(Short_Term_Excitation, self).__init__()

        self.spatical_pooling = nn.AdaptiveAvgPool2d((None, 1))
        self.temporal_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=(5,1), stride=1, bias=False, padding=(2,0), groups=1)
        self.bn = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(in_channels // 4, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_origin = x
        x = self.spatical_pooling(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.expand(x)
        ce_weight = self.sigmoid(x)
        x = x_origin * ce_weight # n,c,t,v * n,c,t,1
        return x

class multi_scale_temporal_excitation(nn.Module):
    def __init__(self, in_channels):
        super(multi_scale_temporal_excitation, self).__init__()
        self.me = Motion_Excitation(in_channels, n_segment=4)
        self.ste = Short_Term_Excitation(in_channels)
        self.lte = Long_Term_Excitation(in_channels)

    def forward(self, x):
        me = self.me(x)
        ste = self.ste(x)
        lte = self.lte(x)
        x = lte + ste + me
        return x