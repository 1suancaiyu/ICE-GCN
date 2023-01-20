import torch.nn as nn

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


class Spatial_Channel_Excitation(nn.Module):
    def __init__(self, out_channels):
        super(Spatial_Channel_Excitation, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>  Spatial_Channel_Excitation")

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        x_origin = x = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)
        x = x.mean(1, keepdim=True)
        x = self.conv(x.view(n,t,1,v).permute(0, 2, 1, 3).contiguous())
        x = x.permute(0, 2, 1, 3).contiguous().view(n * t, 1, v)
        x_spatical_score = self.sigmoid(x)
        x = x_origin * x_spatical_score
        x = x.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()
        return x