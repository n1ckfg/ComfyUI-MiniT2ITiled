import math

import torch
import torch.nn.functional as F

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def _make_base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)


def warp_image(src: torch.Tensor, bwd_flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp src (N, C, H, W) using backward optical flow bwd_flow (N, 2, H, W) in pixel units."""
    orig_dtype = src.dtype
    src_f = src.float()
    flow_f = bwd_flow.to(device=src.device, dtype=torch.float32)
    _, _, h, w = src_f.shape
    base = _make_base_grid(h, w, src.device)[None]  # (1, 2, H, W)
    coords = base + flow_f
    gx = coords[:, 0] * (2.0 / max(w - 1, 1)) - 1.0
    gy = coords[:, 1] * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (N, H, W, 2)
    if mode == "bicubic":
        try:
            warped = F.grid_sample(src_f, grid, mode="bicubic", padding_mode="border", align_corners=True)
        except (NotImplementedError, RuntimeError):
            warped = F.grid_sample(src_f.cpu(), grid.cpu(), mode="bicubic", padding_mode="border", align_corners=True).to(src.device)
        lo = src_f.amin(dim=(-2, -1), keepdim=True)
        hi = src_f.amax(dim=(-2, -1), keepdim=True)
        warped = warped.clamp(lo, hi)
    else:
        warped = F.grid_sample(src_f, grid, mode=mode, padding_mode="border", align_corners=True)
    return warped.to(orig_dtype)


def _blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    k = torch.exp(-(torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype) / sigma) ** 2 / 2)
    k = (k / k.sum()).expand(x.shape[1], 1, -1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.unsqueeze(2), groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.unsqueeze(3), groups=x.shape[1])


def _pytorch_lk_flow(src_gray: torch.Tensor, dst_gray: torch.Tensor, levels: int = 4, iters: int = 5) -> torch.Tensor:
    """Pure-PyTorch multi-scale coarse-to-fine Lucas-Kanade fallback (no OpenCV required).
    Returns flow (1, 2, H, W) mapping pixels in src_gray to dst_gray.
    """
    _, _, h, w = src_gray.shape
    pyr_src = [src_gray]
    pyr_dst = [dst_gray]
    for _ in range(levels - 1):
        if min(pyr_src[-1].shape[-2:]) < 16:
            break
        pyr_src.append(F.avg_pool2d(pyr_src[-1], 2))
        pyr_dst.append(F.avg_pool2d(pyr_dst[-1], 2))

    flow = torch.zeros(1, 2, pyr_src[-1].shape[-2], pyr_src[-1].shape[-1], device=src_gray.device, dtype=torch.float32)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=src_gray.device).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)

    for lvl in range(len(pyr_src) - 1, -1, -1):
        s = pyr_src[lvl]
        d = pyr_dst[lvl]
        lh, lw = s.shape[-2:]
        if flow.shape[-2:] != (lh, lw):
            sh = lh / flow.shape[-2]
            sw = lw / flow.shape[-1]
            flow = F.interpolate(flow, size=(lh, lw), mode="bilinear", align_corners=False)
            flow[:, 0] *= sw
            flow[:, 1] *= sh
        ix_s = F.conv2d(F.pad(s, (1, 1, 1, 1), mode="replicate"), kx)
        iy_s = F.conv2d(F.pad(s, (1, 1, 1, 1), mode="replicate"), ky)
        for _ in range(iters):
            warped_d = warp_image(d, flow)
            it = warped_d - s
            ix = 0.5 * (ix_s + F.conv2d(F.pad(warped_d, (1, 1, 1, 1), mode="replicate"), kx))
            iy = 0.5 * (iy_s + F.conv2d(F.pad(warped_d, (1, 1, 1, 1), mode="replicate"), ky))
            ixx = _blur2d(ix * ix, 2.0) + 1e-4
            iyy = _blur2d(iy * iy, 2.0) + 1e-4
            ixy = _blur2d(ix * iy, 2.0)
            ixt = _blur2d(ix * it, 2.0)
            iyt = _blur2d(iy * it, 2.0)
            det = (ixx * iyy - ixy * ixy).clamp(min=1e-8)
            du = -(iyy * ixt - ixy * iyt) / det
            dv = -(ixx * iyt - ixy * ixt) / det
            delta = torch.cat([du, dv], dim=1).clamp(-4.0, 4.0)
            flow = flow + delta
    return flow


def _calc_pair_flow_cv2(gray_a, gray_b, dis_estimator):
    if dis_estimator is not None:
        flow = dis_estimator.calc(gray_a, gray_b, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    return torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()


def compute_optical_flow(images: torch.Tensor, max_side: int = 1024):
    """Compute backward optical flow (f -> f-1) and soft non-occlusion masks for consecutive frames.

    Both are returned at the estimation resolution (long side at most max_side) so large canvases
    don't keep full-resolution flow per frame; use frame_flow() to get one pair at canvas size.

    Args:
        images: (B, 3, H, W) tensor in [-1, 1], B > 1.
        max_side: Maximum dimension used for flow estimation.

    Returns:
        bwd_flows: (B - 1, 2, h, w) float32 tensor, (dx, dy) in canvas pixels.
        occ_masks: (B - 1, 1, h, w) float32 tensor in [0, 1] (1 = visible in previous frame, 0 = disoccluded/OOB).
    """
    b, _, h, w = images.shape
    device = images.device

    scale = min(1.0, max_side / max(h, w))
    fh = max(16, int(round(h * scale / 2.0)) * 2)
    fw = max(16, int(round(w * scale / 2.0)) * 2)

    # Flow is estimated on the CPU; resizing there also avoids MPS's missing non-integer "area" downscale.
    rgb = (images.detach().float().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    if (fh, fw) != (h, w):
        gray = F.interpolate(gray, size=(fh, fw), mode="area")

    use_cv2 = cv2 is not None and np is not None
    dis = None
    if use_cv2:
        if hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setFinestScale(0)
        gray_np = (gray[:, 0].numpy() * 255.0).round().astype(np.uint8)

    base_small = _make_base_grid(fh, fw, torch.device("cpu"))[None]
    bwd_list = []
    occ_list = []

    for f in range(1, b):
        if use_cv2:
            bwd = _calc_pair_flow_cv2(gray_np[f], gray_np[f - 1], dis)
            fwd = _calc_pair_flow_cv2(gray_np[f - 1], gray_np[f], dis)
        else:
            bwd = _pytorch_lk_flow(gray[f : f + 1], gray[f - 1 : f])
            fwd = _pytorch_lk_flow(gray[f - 1 : f], gray[f : f + 1])

        # Forward-backward consistency error at frame f: ||F_{f->f-1}(p) + F_{f-1->f}(p + F_{f->f-1}(p))||_2
        warped_fwd = warp_image(fwd, bwd)
        err_sq = (bwd + warped_fwd).pow(2).sum(dim=1, keepdim=True)
        mag_sq = bwd.pow(2).sum(dim=1, keepdim=True) + warped_fwd.pow(2).sum(dim=1, keepdim=True)
        # Sundaram et al. threshold (relaxed slightly for smooth soft masking)
        thresh = (0.01 * mag_sq + 0.5).sqrt()
        err = err_sq.clamp(min=0.0).sqrt()
        conf = ((thresh * 1.5 - err) / (thresh * 0.75).clamp(min=0.25)).clamp(0.0, 1.0)

        coords = base_small + bwd
        in_bounds = (
            (coords[:, 0:1] >= 0.0)
            & (coords[:, 0:1] <= fw - 1.0)
            & (coords[:, 1:2] >= 0.0)
            & (coords[:, 1:2] <= fh - 1.0)
        ).float()
        occ = _blur2d(conf * in_bounds, 1.0).clamp(0.0, 1.0)

        bwd_list.append(bwd)
        occ_list.append(occ)

    bwd_flows = torch.cat(bwd_list, dim=0)
    bwd_flows[:, 0] *= w / fw
    bwd_flows[:, 1] *= h / fh
    return bwd_flows.to(device), torch.cat(occ_list, dim=0).to(device)


def frame_flow(bwd_flows: torch.Tensor, occ_masks: torch.Tensor, pair: int, size):
    """Backward flow (1, 2, H, W) and visibility (1, 1, H, W) from frame pair + 1 to frame pair at canvas size,
    with pixels whose source lies outside the canvas marked not visible."""
    h, w = size
    flow = F.interpolate(bwd_flows[pair : pair + 1], size=size, mode="bilinear", align_corners=False)
    occ = F.interpolate(occ_masks[pair : pair + 1], size=size, mode="bilinear", align_corners=False).clamp_(0.0, 1.0)
    coords = _make_base_grid(h, w, flow.device)[None] + flow
    in_bounds = (
        (coords[:, 0:1] >= 0.0)
        & (coords[:, 0:1] <= w - 1.0)
        & (coords[:, 1:2] >= 0.0)
        & (coords[:, 1:2] <= h - 1.0)
    )
    return flow, occ * in_bounds


def warp_noise_sequence(
    b: int,
    channels: int,
    height: int,
    width: int,
    bwd_flows: torch.Tensor,
    occ_masks: torch.Tensor,
    shared_noise: float,
    device: torch.device,
    dtype: torch.dtype,
    generator=None,
) -> torch.Tensor:
    """Generate variance-preserving optical-flow-warped Gaussian noise (B, C, H, W).

    Uses a 2x sub-pixel Gaussian grid with unique-owner nearest advection so:
    - Sub-pixel motions (0.5 px) transport smoothly without bilinear variance loss.
    - Zoom-in collisions and disoccluded regions are refreshed with fresh i.i.d. N(0, 1) noise.
    - 2x2 average pooling * 2.0 produces exact unit-variance N(0, 1) canvas noise on every frame.
    Noise is drawn on the CPU so a seed gives the same noise on every device.
    """
    fh, fw = height * 2, width * 2
    n_fine = fh * fw

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    fine = randn(1, channels, fh, fw)
    out = torch.empty(b, channels, height, width, device=device, dtype=dtype)
    out[0] = F.avg_pool2d(fine, 2)[0] * 2.0
    pos = torch.arange(n_fine, device=device, dtype=torch.int32)
    xs = torch.arange(fw, device=device, dtype=torch.float32)
    ys = torch.arange(fh, device=device, dtype=torch.float32)[:, None]
    # Track per-cell fractional rounding residuals across frames so slow (<0.25 px/frame)
    # and non-0.5-multiple motions accumulate without locking to screen coordinates or slipping.
    rem_x = torch.zeros(fh, fw, device=device, dtype=torch.float32)
    rem_y = torch.zeros(fh, fw, device=device, dtype=torch.float32)

    for f in range(1, b):
        flow = F.interpolate(bwd_flows[f - 1 : f], size=(fh, fw), mode="bilinear", align_corners=False)[0].mul_(2.0)
        visible = F.interpolate(occ_masks[f - 1 : f], size=(fh, fw), mode="bilinear", align_corners=False)[0, 0] > 0.5
        approx_x = (xs + flow[0]).round_().long().clamp_(0, fw - 1)
        approx_y = (ys + flow[1]).round_().long().clamp_(0, fh - 1)
        approx_src = (approx_y * fw + approx_x).flatten()
        total_dx = flow[0].add_(rem_x.flatten()[approx_src].view(fh, fw))
        total_dy = flow[1].add_(rem_y.flatten()[approx_src].view(fh, fw))
        del approx_x, approx_y, approx_src
        step_dx = total_dx.round()
        step_dy = total_dy.round()
        new_rem_x = total_dx.sub_(step_dx)
        new_rem_y = total_dy.sub_(step_dy)
        sx = step_dx.add_(xs)
        sy = step_dy.add_(ys)
        valid = visible & (sx >= 0) & (sx < fw) & (sy >= 0) & (sy < fh)
        src = (sy.long().clamp_(0, fh - 1) * fw + sx.long().clamp_(0, fw - 1)).flatten()
        del flow, sx, sy, visible

        # Resolve zoom-in collisions: at most one valid output sub-pixel (the last one) inherits each source sub-pixel.
        # The rest (collisions, disocclusions, out-of-frame sources) get fresh noise. A full frame is drawn so the
        # random stream doesn't depend on the flow, which differs slightly between devices.
        owner = torch.full((n_fine,), -1, device=device, dtype=torch.int32)
        owner.scatter_reduce_(0, src, torch.where(valid.flatten(), pos, -1), reduce="amax")
        keep = owner[src] == pos
        del owner, valid

        warped = fine.flatten(2)[:, :, src]
        del src
        fresh = randn(1, channels, n_fine)
        fine = torch.where(keep, warped, fresh, out=fresh).view(1, channels, fh, fw)
        keep_2d = keep.view(fh, fw)
        rem_x = torch.where(keep_2d, new_rem_x, torch.zeros_like(new_rem_x))
        rem_y = torch.where(keep_2d, new_rem_y, torch.zeros_like(new_rem_y))
        del warped, keep, keep_2d, new_rem_x, new_rem_y

        coarse = F.avg_pool2d(fine, 2)[0] * 2.0
        if shared_noise < 1.0:
            coarse = math.sqrt(shared_noise) * coarse + math.sqrt(1.0 - shared_noise) * randn(channels, height, width)
        out[f] = coarse

    return out


def anchor_x0(
    x0: torch.Tensor,
    step_idx: int,
    num_steps: int,
    bwd_flows: torch.Tensor,
    occ_masks: torch.Tensor,
    strength: float,
    end_progress: float = 0.80,
):
    """In place: pull each frame's x0 prediction toward the bicubic flow-warped (already anchored) previous frame.

    Cosine-tapers strength to 0 at `end_progress` so later denoising steps sharpen crisp
    high-frequency details while carrying moving textures cleanly across 16x16 patch boundaries.
    """
    progress = step_idx / num_steps
    if progress >= end_progress:
        return
    alpha = 0.85 * strength * 0.5 * (1.0 + math.cos(math.pi * progress / end_progress))
    size = x0.shape[-2:]
    for f in range(1, x0.shape[0]):
        flow, occ = frame_flow(bwd_flows, occ_masks, f - 1, size)
        x0[f : f + 1].lerp_(warp_image(x0[f - 1 : f], flow, mode="bicubic"), alpha * occ)


def temporal_bilateral_stabilize(
    frames: torch.Tensor,
    bwd_flows: torch.Tensor,
    occ_masks: torch.Tensor,
    strength: float,
    sigma_r: float = 0.08,
    max_blend: float = 0.35,
):
    """In place: post-sampling motion-compensated bilateral temporal filter for frames (B, C, H, W) in [-1, 1].

    Damps residual high-frequency pixel shimmer on non-occluded pixels where the bicubic flow-warped
    previous frame closely agrees in color, while rejecting larger differences to avoid ghosting
    or smearing moving grain.
    """
    inv_2sigma2 = 1.0 / (2.0 * sigma_r * sigma_r)
    size = frames.shape[-2:]
    for f in range(1, frames.shape[0]):
        flow, occ = frame_flow(bwd_flows, occ_masks, f - 1, size)
        warped = warp_image(frames[f - 1 : f], flow, mode="bicubic")
        curr = frames[f : f + 1]
        range_w = torch.exp(-(curr - warped).pow(2).mean(dim=1, keepdim=True) * inv_2sigma2)
        curr.lerp_(warped, max_blend * strength * occ * range_w)


def build_tile_flow_grids(
    bwd_flows: torch.Tensor,
    occ_masks: torch.Tensor,
    size,
    tiles,
    tile_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Per-tile patch-grid data (T, F - 1, side, side, 3) for motion-warping frame (f - 1)'s pre-RoPE K/V tokens
    into frame f's coordinates: normalized grid_sample x, y, and the warped token's confidence
    (visible in frame f - 1 and sourced from inside the tile).
    """
    side = tile_size // patch_size
    base = _make_base_grid(side, side, bwd_flows.device)[None]  # (1, 2, side, side)
    scale = 2.0 / (side - 1)
    grids = torch.empty(len(tiles), bwd_flows.shape[0], side, side, 3, device=bwd_flows.device)
    for f in range(bwd_flows.shape[0]):
        flow, occ = frame_flow(bwd_flows, occ_masks, f, size)
        for ti, (y, x0, _) in enumerate(tiles):
            pflow = F.interpolate(flow[:, :, y:y + tile_size, x0:x0 + tile_size], size=(side, side), mode="area") / patch_size
            grid = (base + pflow) * scale - 1.0
            inside = (grid.abs() <= 1.0).all(dim=1)
            grids[ti, f, ..., :2] = grid[0].permute(1, 2, 0)
            grids[ti, f, ..., 2] = F.interpolate(occ[:, :, y:y + tile_size, x0:x0 + tile_size], size=(side, side), mode="area")[0, 0] * inside[0]
    return grids
