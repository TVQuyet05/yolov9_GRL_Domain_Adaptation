import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

def SpectralConv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def SpectralLinear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class DiscriminatorHead(nn.Module):
    """
    Phần đầu của bộ phân loại. Nó nhận các đặc trưng và dự đoán miền của chúng.
    Được thiết kế với nhiều lớp tuyến tính, sử dụng chuẩn hóa phổ để cải thiện
    sự ổn định khi huấn luyện.
    """
    def __init__(self, dim_in, dim_h, dim_o=1):
        super().__init__()
        self.to_flat = nn.Sequential(
            SpectralConv2d(dim_in, dim_h // 2, kernel_size=1),
            nn.Flatten(),
            nn.LazyLinear(dim_h),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.neck = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_h // 2, dim_h // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(3)
        ])
        self.head = nn.Sequential(
            SpectralLinear(dim_h // 2 * 4, dim_h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim_h // 2, dim_o, bias=False),
        )

    def forward(self, x):
        x = self.to_flat(x)
        x = x.split(x.shape[1] // 2, dim=1)
        xs = [x[0]]
        for m in self.neck:
            x = m(x[1]) if isinstance(x, tuple) else m(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        return self.head(x)

class Discriminator(nn.Module):
    """
    Bộ phân loại (Discriminator) chính.
    Bao gồm lớp đảo ngược gradient và các lớp tích chập để xử lý các đặc trưng
    từ mô hình YOLOv8 và đưa chúng vào DiscriminatorHead.
    """
    def __init__(self, chs=None, amp=False):
        super().__init__()
        if chs is None:
            chs = [256] # P4 feature map channels for Yolov9s? P4 out channel in yolov9-s is 192 (from layer 6), but let's just make it configurable. 
        self.chs = chs
        self.f_len = len(chs)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.amp = amp
        # Các lớp để trích xuất đặc trưng
        self.p = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i] if i == 0 else chs[i] * 2, 64, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, chs[i + 1] if i + 1 < len(chs) else chs[i], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(chs[i + 1] if i + 1 < len(chs) else chs[i]),
                nn.SiLU(inplace=True),
            ) for i in range(len(chs))
        ])
        self.head = DiscriminatorHead(chs[-1], 256)
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, fs: list):
        with torch.cuda.amp.autocast(self.amp):
            assert len(fs) == self.f_len, f'Expected {self.f_len} feature maps, got {len(fs)}'
            fs = [self.grl(f) for f in fs] # Áp dụng lớp đảo ngược gradient
            x = self.p[0](fs[0])
            for i in range(1, len(fs)):
                x = torch.cat((x, fs[i]), dim=1)
                x = self.p[i](x)
            return self.head(x)