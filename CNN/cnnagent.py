import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # needed if you're going to check isinstance(x, np.ndarray)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, activation=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = activation()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        hidden_ch = out_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden_ch, kernel_size=1, stride=1, padding=0)
        self.conv_blocks = nn.Sequential(
            *[ConvBNAct(hidden_ch, hidden_ch) for _ in range(n)]
        )
        self.conv2 = ConvBNAct(in_ch, hidden_ch, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBNAct(2 * hidden_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.conv_blocks(y1)
        y2 = self.conv2(x)
        out = torch.cat([y1, y2], dim=1)
        out = self.conv3(out)
        return out


class FeatureExtractor(nn.Module):
    """
    A YOLOv5n-like backbone that:
      1) Optionally resizes input to [600, 600]
      2) Processes it through CSP blocks
      3) Outputs a flattened feature vector of size [batch_size, 256]
      4) Can handle NumPy inputs and move them to the specified device
    """
    def __init__(self, device=None):
        super().__init__()
        # If no device specified, pick GPU if available, else CPU
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stem = nn.Sequential(
            ConvBNAct(3, 16, kernel_size=3, stride=2, padding=1),
            ConvBNAct(16, 16, kernel_size=3, stride=1, padding=1),
        )

        self.csp1 = CSPBlock(16, 32, n=1)
        self.down1 = ConvBNAct(32, 32, kernel_size=3, stride=2, padding=1)
        self.csp2 = CSPBlock(32, 64, n=1)
        self.down2 = ConvBNAct(64, 64, kernel_size=3, stride=2, padding=1)
        self.csp3 = CSPBlock(64, 128, n=1)
        self.down3 = ConvBNAct(128, 128, kernel_size=3, stride=2, padding=1)
        self.csp4 = CSPBlock(128, 256, n=1)

        self.final_conv = ConvBNAct(256, 256, kernel_size=1, stride=1, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Move entire model to the chosen device
        self.to(self.device)

    def forward(self, x):
        """
        x: could be a torch.Tensor or a np.ndarray
           shape expected: [batch_size, 3, H, W]
           If you have [batch_size, H, W, 3] for images in NumPy, 
           you'll need to permute dimensions.

        Returns: [batch_size, 256]
        """
        # 1) If x is NumPy, convert to torch.Tensor
        if isinstance(x, np.ndarray):
            # If your NumPy images are [B, H, W, C], permute them first:
            # x = np.transpose(x, (0, 3, 1, 2))
            x = torch.from_numpy(x).float()

        # 2) Move x to device
        x = x.to(self.device)
        if x.ndim == 3:
            # Assuming shape is [H, W, C]
            # Permute to [C, H, W] and unsqueeze batch dim -> [1, C, H, W]
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 4:
            # Assuming shape is [N, H, W, C]
            # Permute to [N, C, H, W]
            x = x.permute(0, 3, 1, 2)


        # (Optional) Resize to 600 x 600
        # x = F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False)

        # 3) Pass through the YOLO-like stem & CSP blocks
        x = self.stem(x)     
        x = self.csp1(x)     
        x = self.down1(x)    
        x = self.csp2(x)     
        x = self.down2(x)    
        x = self.csp3(x)     
        x = self.down3(x)    
        x = self.csp4(x)     
        x = self.final_conv(x)

        # 4) Global average pool -> [B, 256, 1, 1]
        x = self.global_pool(x)

        # 5) Flatten -> [B, 256]
        x = x.view(x.size(0), -1)
        return x


# Example usage
if __name__ == "__main__":
    # Example 1: Torch Tensor
    model = FeatureExtractor()  # auto device selection
    dummy_torch = torch.randn(2, 3, 720, 1300)  # [batch=2, 3, H,
