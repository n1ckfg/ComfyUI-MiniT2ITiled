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


def warp_image(src: torch.Tensor, bwd_flow: torch.Tensor) -> torch.Tensor:
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
    warped = F.grid_sample(src_f, grid, mode="bilinear", padding_mode="border", align_corners=True)
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


def compute_optical_flow(images: torch.Tensor, max_side: int = 512):
    """Compute backward optical flow (f -> f-1) and soft non-occlusion masks for consecutive frames.

    Args:
        images: (B, 3, H, W) tensor in [-1, 1].
        max_side: Maximum dimension used for flow estimation before upscaling to (H, W).

    Returns:
        bwd_flows: (B - 1, 2, H, W) float32 tensor in full-resolution pixel units (dx, dy).
        occ_masks: (B - 1, 1, H, W) float32 tensor in [0, 1] (1 = visible in previous frame, 0 = disoccluded/OOB).
    """
    b, _, h, w = images.shape
    device = images.device
    if b <= 1:
        return (
            torch.zeros(0, 2, h, w, device=device, dtype=torch.float32),
            torch.zeros(0, 1, h, w, device=device, dtype=torch.float32),
        )

    scale = min(1.0, max_side / max(h, w))
    fh = max(16, int(round(h * scale / 2.0)) * 2)
    fw = max(16, int(round(w * scale / 2.0)) * 2)

    rgb = (images.detach().float() * 0.5 + 0.5).clamp(0.0, 1.0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    if (fh, fw) != (h, w):
        gray = F.interpolate(gray, size=(fh, fw), mode="area")

    use_cv2 = cv2 is not None and np is not None
    dis = None
    if use_cv2:
        if hasattr(cv2, "DISOpticalFlow_create"):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        gray_np = (gray[:, 0].cpu().numpy() * 255.0).round().astype(np.uint8)

    base_small = _make_base_grid(fh, fw, torch.device("cpu"))[None]
    bwd_list = []
    occ_list = []

    for f in range(1, b):
        if use_cv2:
            bwd = _calc_pair_flow_cv2(gray_np[f], gray_np[f - 1], dis)
            fwd = _calc_pair_flow_cv2(gray_np[f - 1], gray_np[f], dis)
        else:
            s_cur = gray[f : f + 1].cpu()
            s_prev = gray[f - 1 : f].cpu()
            bwd = _pytorch_lk_flow(s_cur, s_prev)
            fwd = _pytorch_lk_flow(s_prev, s_cur)

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

    bwd_flows = torch.cat(bwd_list, dim=0).to(device)
    occ_masks = torch.cat(occ_list, dim=0).to(device)

    if (fh, fw) != (h, w):
        bwd_flows = F.interpolate(bwd_flows, size=(h, w), mode="bilinear", align_corners=False)
        bwd_flows[:, 0] *= w / fw
        bwd_flows[:, 1] *= h / fh
        occ_masks = F.interpolate(occ_masks, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    # Re-enforce strict full-res border mask
    base_full = _make_base_grid(h, w, device)[None]
    coords_full = base_full + bwd_flows
    in_bounds_full = (
        (coords_full[:, 0:1] >= 0.0)
        & (coords_full[:, 0:1] <= w - 1.0)
        & (coords_full[:, 1:2] >= 0.0)
        & (coords_full[:, 1:2] <= h - 1.0)
    ).float()
    occ_masks = occ_masks * in_bounds_full

    return bwd_flows, occ_masks


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
    """
    s = min(max(float(shared_noise), 0.0), 1.0)
    fh, fw = height * 2, width * 2
    fine = torch.randn(1, channels, fh, fw, generator=generator, device=device, dtype=torch.float32)
    frames = [F.avg_pool2d(fine, 2) * 2.0]

    base_fine = _make_base_grid(fh, fw, device)  # (2, 2H, 2W)
    n_fine = fh * fw

    for f in range(1, b):
        flow_f = F.interpolate(bwd_flows[f - 1 : f].float(), size=(fh, fw), mode="bilinear", align_corners=False)[0] * 2.0
        occ_f = F.interpolate(occ_masks[f - 1 : f].float(), size=(fh, fw), mode="bilinear", align_corners=False)[0, 0]

        sx = (base_fine[0] + flow_f[0]).round().long()
        sy = (base_fine[1] + flow_f[1]).round().long()
        valid = (occ_f > 0.5) & (sx >= 0) & (sx < fw) & (sy >= 0) & (sy < fh)
        src_idx = (sy.clamp(0, fh - 1) * fw + sx.clamp(0, fw - 1)).reshape(-1)
        flat_valid = valid.reshape(-1)

        # Resolve zoom-in collisions: at most one output sub-pixel may inherit each source sub-pixel.
        keep = torch.zeros(n_fine, dtype=torch.bool, device=device)
        valid_pos = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if valid_pos.numel() > 0:
            owner = torch.full((n_fine,), -1, dtype=torch.long, device=device)
            src_valid = src_idx[valid_pos]
            owner[src_valid] = valid_pos
            keep[valid_pos] = owner[src_valid] == valid_pos

        fresh_fine = torch.randn(1, channels, fh, fw, generator=generator, device=device, dtype=torch.float32)
        warped_fine = fine.reshape(1, channels, n_fine)[:, :, src_idx].reshape(1, channels, fh, fw)
        fine = torch.where(keep.reshape(1, 1, fh, fw), warped_fine, fresh_fine)

        coarse = F.avg_pool2d(fine, 2) * 2.0
        if s < 1.0:
            fresh_coarse = torch.randn(1, channels, height, width, generator=generator, device=device, dtype=torch.float32)
            coarse = math.sqrt(s) * coarse + math.sqrt(1.0 - s) * fresh_coarse
        frames.append(coarse)

    return torch.cat(frames, dim=0).to(dtype)


def anchor_x0_velocity(
    x: torch.Tensor,
    v: torch.Tensor,
    t_cur: torch.Tensor,
    step_idx: int,
    num_steps: int,
    bwd_flows: torch.Tensor,
    occ_masks: torch.Tensor,
    flow_consistency: float,
    end_progress: float = 0.65,
) -> torch.Tensor:
    """Anchor early-step CFG x0 predictions across consecutive frames along optical flow.

    Cosine-tapers strength to 0 at `end_progress` so later denoising steps sharpen crisp
    high-frequency details without accumulating bilinear warp blur.
    """
    strength = min(max(float(flow_consistency), 0.0), 1.0)
    b = x.shape[0]
    if strength <= 0.0 or b <= 1:
        return v
    progress = step_idx / max(num_steps, 1)
    if progress >= end_progress:
        return v

    decay = 0.5 * (1.0 + math.cos(math.pi * progress / end_progress))
    alpha = 0.85 * strength * decay
    if alpha <= 1e-4:
        return v

    remaining = torch.clamp(1.0 - t_cur.float(), min=0.001)[:, None, None, None]
    x0_cfg = x.float() + remaining * v.float()
    x0_anchored = x0_cfg.clone()

    for f in range(1, b):
        warped_prev = warp_image(x0_anchored[f - 1 : f], bwd_flows[f - 1 : f])
        w = alpha * occ_masks[f - 1 : f]
        x0_anchored[f : f + 1] = x0_anchored[f : f + 1] + w * (warped_prev - x0_anchored[f : f + 1])

    delta_v = (x0_anchored - x0_cfg) / remaining
    return v + delta_v.to(v.dtype)


def temporal_bilateral_stabilize(
    frames: torch.Tensor,
    bwd_flows: torch.Tensor,
    occ_masks: torch.Tensor,
    strength: float,
    sigma_r: float = 0.12,
    max_blend: float = 0.4,
) -> torch.Tensor:
    """Post-sampling motion-compensated bilateral temporal filter for frames (B, C, H, W) in [-1, 1].

    Damps residual high-frequency pixel shimmer on non-occluded pixels where the flow-warped
    previous frame closely agrees in color, while rejecting large differences to avoid ghosting
    or edge blur.
    """
    s = min(max(float(strength), 0.0), 1.0)
    b = frames.shape[0]
    if s <= 0.0 or b <= 1:
        return frames

    orig_dtype = frames.dtype
    out = frames.float().clone()
    inv_2sigma2 = 1.0 / (2.0 * sigma_r * sigma_r)
    alpha = max_blend * s

    for f in range(1, b):
        warped_prev = warp_image(out[f - 1 : f], bwd_flows[f - 1 : f])
        curr = out[f : f + 1]
        diff_sq = (curr - warped_prev).pow(2).mean(dim=1, keepdim=True)
        range_w = torch.exp(-diff_sq * inv_2sigma2)
        w = alpha * occ_masks[f - 1 : f] * range_w
        out[f : f + 1] = curr + w * (warped_prev - curr)

    return out.to(orig_dtype)


def build_tile_flow_grids(
    bwd_flows: torch.Tensor,
    tiles,
    tile_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Build normalized grid_sample coordinates (T, F - 1, side, side, 2) on each tile's patch grid
    so frame (f - 1)'s pre-RoPE K/V tokens can be motion-warped into frame f's coordinates.
    """
    side = tile_size // patch_size
    base = _make_base_grid(side, side, bwd_flows.device)[None]  # (1, 2, side, side)
    scale_x = 2.0 / max(side - 1, 1)
    scale_y = 2.0 / max(side - 1, 1)
    grids = []
    for y, x0, _ in tiles:
        tile_flow = bwd_flows[:, :, y:y + tile_size, x0:x0 + tile_size].float()
        pflow = F.interpolate(tile_flow, size=(side, side), mode="area") / float(patch_size)
        coords = base + pflow
        gx = coords[:, 0] * scale_x - 1.0
        gy = coords[:, 1] * scale_y - 1.0
        grids.append(torch.stack([gx, gy], dim=-1))
    return torch.stack(grids, dim=0)  # (T, F - 1, side, side, 2)


