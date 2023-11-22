import torch

class litedetect(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1
        self.a = Conv(in_channels=3, out_channels=8, kernel_size=1, stride=2) # down to 320x320x8
        self.b = Conv(in_channels=8, out_channels=16, kernel_size=1, stride=2) # down to 160x160x16
        self.c = Conv(in_channels=16, out_channels=32, kernel_size=1, stride=2) # down to 80x80x32
        self.d = Conv(in_channels=32, out_channels=64, kernel_size=1, stride=2) # down to 40x40x64
        self.e = Conv(in_channels=64, out_channels=64, kernel_size=1, stride=2) # down to 20x20x64
        self.f = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(1,1), padding=(1,1)) # still 20x20x64

        # Layer 2
        self.f_up = torch.nn.Upsample(size=(40,40)) # 40x40x64
        self.d_l2 = Conv(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.d_l2_up = torch.nn.Upsample(size=(80,80)) # 80x80x128
        self.c_l2 = Conv(in_channels=160, out_channels=160, kernel_size=1, stride=1)


    def forward(self, x):
        # Layer 1
        a_out = self.a(x)
        b_out = self.b(a_out)
        c_out = self.c(b_out)
        d_out = self.d(c_out)
        e_out = self.e(d_out)
        f_out = self.f(e_out)

        # Layer 2
        f_up_out = self.f_up(f_out) # 40x40x64
        f_d_cat = torch.concat(f_up_out, d_out) # 40x40x128
        d_l2_out = self.d_l2(f_d_cat)
        d_l2_up_out = self.d_l2_up(d_l2_out) # 80x80x128
        d_c_cat = torch.concat(d_l2_up_out, c_out) # 80x80x160
        c_l2_out = self.c_l2(d_c_cat)


class Conv(torch.nn.Module):
    def  __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.a = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)
        self.b = torch.nn.BatchNorm2d(num_features=out_channels)
        self.c = torch.nn.SiLU()

    def forward(self,x):
        a_out = self.a(x)
        b_out = self.b(a_out)
        return self.c(b_out)
