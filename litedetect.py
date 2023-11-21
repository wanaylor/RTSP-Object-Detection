import torch

class litedetect(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.a = self.Conv(in_channels=3, out_channels=8, kernel_size=1, stride=2) # down to 320x320
        self.b = self.Conv(in_channels=8, out_channels=16, kernel_size=1, stride=2) # down to 160x160
        self.c = self.Conv(in_channels=16, out_channels=8, kernel_size=1, stride=2) # down to 80x80
        self.d = self.Conv(in_channels=8, out_channels=4, kernel_size=1, stride=2) # down to 40x40

        self.a_pooled = torch.nn.MaxPool2d(kernel_size=(2,2)) # down to 160x160
        self.b_pooled = torch.nn.MaxPool2d(kernel_size=(2,2)) # down to 80x80
        self.c_pooled = torch.nn.MaxPool2d(kernel_size=(2,2)) # down to 40x40
        self.d_pooled = torch.nn.MaxPool2d(kernel_size=(2,2)) # down to 20x20


    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(a_out)
        c_out = self.c(b_out)
        d_out = self.d(c_out)

        a_pooled = torch.nn.MaxPool2d(kernel_size=(2,2))

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
