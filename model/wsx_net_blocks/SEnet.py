import torch.nn as nn

# skeleton SEnet
class SEnet(nn.Module):
    def __init__(self, in_channels, stride=1, sq_scale=16, n_segment=4):
        super(SEnet, self).__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.stride = stride

        if in_channels == 3:
            self.reduced_channels = 3
        else:
            self.reduced_channels = self.in_channels // sq_scale

        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.action_p2_squeeze = nn.Conv2d(in_channels=self.in_channels, out_channels=self.reduced_channels, kernel_size=1)
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.action_p2_expand = nn.Conv2d(in_channels=self.reduced_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        # get origin
        n, c, t, v = x.size()
        n_batch = n * t // self.n_segment
        x_origin = x.permute(0, 2, 1, 3).contiguous().view(n * t, c, v)

        # 2D convolution: c*T*1*1, channel excitation
        x_ce = self.avg_pool(x) # spatial pooling
        x_ce = self.action_p2_squeeze(x_ce)

        _, c_r, _, _ = x_ce.size()
        # x_ce = x_ce.view(n_batch, self.n_segment, c_r, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        # x_ce = self.action_p2_conv1(x_ce)
        x_ce = self.relu(x_ce)

        # reshape
        # x_ce = x_ce.transpose(2,1).contiguous().view(-1, c_r, 1)

        # expand
        # x_ce = x_ce.view(n, t, c_r, 1).permute(0, 2, 1, 3).contiguous()
        x_ce = self.action_p2_expand(x_ce)
        x_ce = self.sigmoid(x_ce)
        x_ce = x_ce.permute(0, 2, 1, 3).contiguous().view(n * t, c, 1)

        # merge
        x_ce = x_origin * x_ce + x_origin

        # out
        x_ce = x_ce.view(n, t, c, v).permute(0, 2, 1, 3).contiguous()

        return x_ce