import itertools
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call

from comfy.ldm.depth_anything_3.preprocess import compute_target_size

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gray(img):
    w = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
    return (img * w[None, :, None, None]).sum(1, keepdim=True)


def blur(x, sigma):
    radius = math.ceil(3 * sigma)
    k = torch.exp(-(torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype) / sigma) ** 2 / 2)
    k = (k / k.sum()).expand(x.shape[1], 1, -1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.unsqueeze(2), groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.unsqueeze(3), groups=x.shape[1])


def sobel_xy(x):
    k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x, k.view(1, 1, 3, 3)), F.conv2d(x, k.t().reshape(1, 1, 3, 3))


def sobel(x):
    gx, gy = sobel_xy(x)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def correlation(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def downscale(x, factor):
    # Gaussian prefilter + stride: differentiable on every backend (antialiased interpolate has no MPS backward)
    return blur(x, factor / 2)[..., ::factor, ::factor]


class EdgeControl:
    # target: control map in [0, 1], white edges on black, (B, C, H, W) at canvas size.
    # Edges are compared at reduced resolution so the gradient moves shapes instead of drawing lines.
    factor = 4
    sigma = 1.5

    def __init__(self, control, target):
        self.strength, self.start, self.end = control["strength"], control["start"], control["end"]
        self.target = blur(downscale(gray(target), self.factor), self.sigma)

    def loss(self, x):
        return -correlation(blur(sobel(downscale(gray(x * 0.5 + 0.5), self.factor)), self.sigma), self.target)


class DepthControl:
    # target: depth map in [0, 1], near = white. Depth Anything 3 estimates depth from the x0 prediction;
    # correlation makes the match independent of the target's depth scale and offset.
    def __init__(self, control, target):
        self.strength, self.start, self.end = control["strength"], control["start"], control["end"]
        self.net = control["depth_model"].model.diffusion_model
        self.dtype = self.net.dtype or torch.float32
        # ComfyUI loads weights under inference_mode, which autograd can't save; clone for this call only.
        self.params = {k: p.clone() for k, p in itertools.chain(self.net.named_parameters(), self.net.named_buffers())}
        self.size = compute_target_size(target.shape[-2], target.shape[-1], 504)
        self.target = F.interpolate(gray(target), size=self.size, mode="bilinear", antialias=True)
        self.mean = torch.tensor(IMAGENET_MEAN, device=target.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=target.device).view(1, 3, 1, 1)

    def depth(self, x):
        x = (x * 0.5 + 0.5).clamp(0, 1)
        factor = x.shape[-1] / self.size[1]
        if factor > 1.5:
            x = blur(x, factor / 2)
        x = (F.interpolate(x, size=self.size, mode="bilinear") - self.mean) / self.std
        return functional_call(self.net, self.params, (x.to(self.dtype),))["depth"].float()

    def loss(self, x):
        return -correlation(-self.depth(x), self.target)


class NormalControl(DepthControl):
    # target: normal map in the ControlNet/BAE convention, R = facing left, G = facing up, B = facing the viewer.
    # Compares those with the slopes of the estimated depth, which carries the same orientation up to scale.
    def __init__(self, control, target):
        super().__init__(control, target)
        self.target = blur(F.interpolate(target[:, :2], size=self.size, mode="bilinear", antialias=True) * 2 - 1, 2.0)

    def loss(self, x):
        slope_x, slope_y = sobel_xy(blur(-self.depth(x).unsqueeze(1), 2.0))
        return -(correlation(slope_x, self.target[:, :1]) + correlation(slope_y, self.target[:, 1:])) / 2


CONTROL_TYPES = {"edge": EdgeControl, "depth": DepthControl, "normal": NormalControl}


def rms(x):
    return x.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp(min=1e-12)


class Guide:
    # Correction to subtract from the x0 prediction. Control losses are differentiated at the conditional
    # x0 prediction, then pulled back through the model to x_t with vjp, so the model turns the requested
    # change into image structure instead of painted-on patterns.
    # ComfyUI runs nodes under inference_mode, which has to be disabled for autograd,
    # including for the targets that the losses save for backward.
    def __init__(self, controls, height, width, device):
        with torch.inference_mode(False):
            self.controls = []
            for c in controls:
                target = c["image"][..., :3].movedim(-1, 1).to(device, torch.float32)
                if target.shape[-2:] != (height, width):
                    target = F.interpolate(target, size=(height, width), mode="bilinear", antialias=True)
                self.controls.append(CONTROL_TYPES[c["type"]](c, target))

    def active(self, t):
        return any(c.start <= t <= c.end for c in self.controls)

    def correction(self, x0, t, vjp):
        with torch.inference_mode(False), torch.enable_grad():
            x = x0.detach().clone().float().requires_grad_(True)
            grad_x0 = torch.zeros_like(x)
            for c in self.controls:
                if c.start <= t <= c.end:
                    grad, = torch.autograd.grad(c.loss(x).sum(), x)
                    grad_x0 += c.strength * grad / rms(grad)
            grad_xt = vjp(grad_x0.to(x0.dtype)).float()
            grad_xt = grad_xt.mean(1, keepdim=True)  # same change on every channel: move brightness, not hue
            return (grad_xt * (rms(grad_x0) / rms(grad_xt))).to(x0.dtype)
