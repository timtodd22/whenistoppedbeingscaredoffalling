"""Real-ESRGAN (realesr-general-x4v3, SRVGGNetCompact) on CPU, no basicsr dependency."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(__import__('os').environ.get('SR_THREADS', '4')))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1)); self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


_net = None


def net():
    global _net
    if _net is None:
        _net = SRVGGNetCompact()
        sd = torch.load('weights/realesr-general-x4v3.pth', map_location='cpu', weights_only=True)
        _net.load_state_dict(sd.get('params', sd)); _net.eval()
    return _net


@torch.inference_mode()
def upscale4(rgb):
    """uint8 HxWx3 -> uint8 4Hx4Wx3"""
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
    y = net()(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return (y * 255 + 0.5).astype(np.uint8)
