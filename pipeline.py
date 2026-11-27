import itertools
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.func import functional_call
import torch.nn.functional as F

try:
    from .temporal import anchor_x0_velocity, compute_optical_flow, warp_noise_sequence
except ImportError:
    from temporal import anchor_x0_velocity, compute_optical_flow, warp_noise_sequence


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def rotate_half(x):
    x1, x2 = x.reshape(*x.shape[:-1], 2, -1).unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_channels=3, pca_channels=128, hidden_size=1248):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        hidden_dim = (hidden_features + 7) // 8 * 8
        self.w1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w3 = nn.Linear(in_features, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


_TEXT_ROPE_CACHE = {}
_VISION_ROPE_CACHE = {}
_POS_EMBED_CACHE = {}
_SCHEDULER_CACHE = {}


def _cache_put(cache: dict, key, value, max_size: int = 64):
    if len(cache) >= max_size:
        cache.pop(next(iter(cache)))
    cache[key] = value


class TextRotaryEmbedding1D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x):
        _, length, _, d = x.shape
        key = (length, d, self.theta, x.device, x.dtype)
        cached = _TEXT_ROPE_CACHE.get(key)
        if cached is None:
            with torch.inference_mode(False), torch.no_grad():
                inv = 1.0 / (self.theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
                pos = torch.arange(length, device=x.device, dtype=torch.float32)
                angles = torch.einsum("l,f->lf", pos, inv)
                angles = torch.cat([angles, angles], dim=-1)
                cos = angles.cos().to(dtype=x.dtype)[None, :, None, :]
                sin = angles.sin().to(dtype=x.dtype)[None, :, None, :]
                cached = (cos, sin)
                _cache_put(_TEXT_ROPE_CACHE, key, cached)
        cos, sin = cached
        return x * cos + rotate_half(x) * sin


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = head_dim // 2
        self.theta = theta

    def forward(self, x):
        length = x.shape[1]
        key = (length, self.dim, self.theta, x.device, x.dtype)
        cached = _VISION_ROPE_CACHE.get(key)
        if cached is None:
            side = int(math.sqrt(length))
            if side * side != length:
                raise ValueError(f"image token length must be square, got {length}")
            with torch.inference_mode(False), torch.no_grad():
                freqs = 1.0 / (
                    self.theta
                    ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)[: self.dim // 2] / self.dim)
                )
                t = torch.arange(side, device=x.device, dtype=torch.float32)
                base = torch.einsum("l,f->lf", t, freqs)
                f_h, f_w = torch.broadcast_tensors(base[:, None, :], base[None, :, :])
                angles = torch.cat([f_h, f_w], dim=-1)
                angles = torch.cat([angles, angles], dim=-1).reshape(length, -1)
                cos = angles.cos().to(dtype=x.dtype)[None, :, None, :]
                sin = angles.sin().to(dtype=x.dtype)[None, :, None, :]
                cached = (cos, sin)
                _cache_put(_VISION_ROPE_CACHE, key, cached)
        cos, sin = cached
        return x * cos + rotate_half(x) * sin


class MultiModalRotaryEmbeddingFast(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.text_rope = TextRotaryEmbedding1D(head_dim)
        self.vision_rope = VisionRotaryEmbeddingFast(head_dim)

    def forward(self, x, txt_len: int):
        txt = self.text_rope(x[:, :txt_len])
        img = self.vision_rope(x[:, txt_len:])
        return torch.cat([txt, img], dim=1)


class PlainTextTransformerBlock(nn.Module):
    def __init__(self, hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.attn_proj = nn.Linear(inner_dim, hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = TextRotaryEmbedding1D(head_dim)

    def forward(self, txt):
        b, length, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.rope(self.q_norm(q)).transpose(1, 2)
        k = self.rope(self.k_norm(k)).transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, -1)
        txt = txt + self.attn_proj(out)
        txt = txt + self.mlp(self.norm2(txt))
        return txt


class DoubleStreamDiTBlock(nn.Module):
    def __init__(self, hidden_size=1248, txt_hidden_size=1248, num_heads=24, head_dim=52, mlp_ratio=2.7):
        super().__init__()
        self.hidden_size = hidden_size
        self.txt_hidden_size = txt_hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(txt_hidden_size)
        self.txt_norm2 = RMSNorm(txt_hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.txt_qkv = nn.Linear(txt_hidden_size, inner_dim * 3)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = MultiModalRotaryEmbeddingFast(head_dim)
        self.img_attn_proj = nn.Linear(inner_dim, hidden_size)
        self.txt_attn_proj = nn.Linear(inner_dim, txt_hidden_size)
        self.img_mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLUMlp(txt_hidden_size, int(txt_hidden_size * mlp_ratio))

    def forward(self, x, txt, vec, temporal_frames: int = 1, temporal_bias=None):
        b, li, _ = x.shape
        lt = txt.shape[1]
        x_norm = self.img_norm1(x)
        txt_norm = self.txt_norm1(txt)
        qkv_i = self.img_qkv(x_norm).reshape(b, li, 3, self.num_heads, self.head_dim)
        qkv_t = self.txt_qkv(txt_norm).reshape(b, lt, 3, self.num_heads, self.head_dim)
        q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
        q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
        q_i, k_i = self.q_norm(q_i), self.k_norm(k_i)
        q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)
        q = self.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt).transpose(1, 2)
        k = self.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt).transpose(1, 2)
        v = torch.cat([v_t, v_i], dim=1).transpose(1, 2)
        if temporal_frames > 1 and temporal_bias is not None and b % temporal_frames == 0:
            groups = b // temporal_frames
            k_img = k[:, :, lt:].reshape(groups, temporal_frames, self.num_heads, li, self.head_dim)
            v_img = v[:, :, lt:].reshape(groups, temporal_frames, self.num_heads, li, self.head_dim)
            k_anchor = k_img[:, :1].expand(groups, temporal_frames, self.num_heads, li, self.head_dim).reshape(b, self.num_heads, li, self.head_dim)
            v_anchor = v_img[:, :1].expand(groups, temporal_frames, self.num_heads, li, self.head_dim).reshape(b, self.num_heads, li, self.head_dim)
            if temporal_frames == 2:
                k = torch.cat([k, k_anchor], dim=2)
                v = torch.cat([v, v_anchor], dim=2)
            else:
                k_prev = torch.cat([k_img[:, :1], k_img[:, :-1]], dim=1).reshape(b, self.num_heads, li, self.head_dim)
                v_prev = torch.cat([v_img[:, :1], v_img[:, :-1]], dim=1).reshape(b, self.num_heads, li, self.head_dim)
                k = torch.cat([k, k_anchor, k_prev], dim=2)
                v = torch.cat([v, v_anchor, v_prev], dim=2)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=temporal_bias).transpose(1, 2)
        else:
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        x = x + self.img_attn_proj(out[:, lt:].reshape(b, li, -1))
        txt = txt + self.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
        x = x + self.img_mlp(self.img_norm2(x))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return x, txt


class FinalLayer(nn.Module):
    def __init__(self, hidden_size=1248, patch_size=16, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, vec=None):
        return self.linear(self.norm_final(x))


def get_2d_sincos_pos_embed(embed_dim, grid_size, device, dtype):
    key = (embed_dim, grid_size, device, dtype)
    cached = _POS_EMBED_CACHE.get(key)
    if cached is None:
        with torch.inference_mode(False), torch.no_grad():
            grid_h = torch.arange(grid_size, device=device, dtype=torch.float32)
            grid_w = torch.arange(grid_size, device=device, dtype=torch.float32)
            grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
            grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
            emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
            emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
            cached = torch.cat([emb_h, emb_w], dim=1).to(dtype=dtype)
            _cache_put(_POS_EMBED_CACHE, key, cached)
    return cached


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


@dataclass
class MMJiTConfig:
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 3
    txt_input_size: int = 1024
    hidden_size: int = 768
    txt_hidden_size: int = 768
    cond_vec_size: int = 768
    depth_double: int = 17
    txt_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128
    prompt_length: int = 256
    n_T: int = 100
    prediction: str = "x"
    sampler: str = "euler"
    cfg_channels: int = 3
    cfg_interval: tuple = (0.0, 1.0)
    llm: str = "google/flan-t5-large"


class MMJiT(nn.Module):
    def __init__(self, cfg: MMJiTConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_img_size = cfg.image_size // cfg.patch_size
        self.img_embedder = BottleneckPatchEmbed(
            cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.pca_channels, cfg.hidden_size
        )
        self.txt_embedder = nn.Linear(cfg.txt_input_size, cfg.txt_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.txt_input_size))
        self.t_embedder = TimestepEmbedder(cfg.cond_vec_size)
        self.pooled_embedder = nn.Linear(cfg.txt_input_size, cfg.cond_vec_size, bias=False)
        self.txt_preamble_blocks = nn.ModuleList(
            [PlainTextTransformerBlock(cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio) for _ in range(cfg.txt_preamble_depth)]
        )
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamDiTBlock(
                    cfg.hidden_size, cfg.txt_hidden_size, cfg.num_heads, cfg.head_dim, cfg.mlp_ratio
                )
                for _ in range(cfg.depth_double)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.in_channels)

    def unpatchify(self, x):
        b = x.shape[0]
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(b, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(b, c, h * p, w * p)

    def prepare_text(self, context, attn_mask):
        attn_mask = attn_mask.to(device=context.device)
        context = torch.where(attn_mask[:, :, None] > 0.5, context, self.mask_token.to(dtype=context.dtype))
        txt = self.txt_embedder(context.to(dtype=self.txt_embedder.weight.dtype))
        pooled_text = context.mean(dim=1)
        pooled_vec = self.pooled_embedder(pooled_text.to(dtype=self.pooled_embedder.weight.dtype))
        for block in self.txt_preamble_blocks:
            txt = block(txt)
        return txt, pooled_vec

    def forward(self, img, t, context, attn_mask, prepared_text=None, temporal_frames: int = 1, temporal_consistency: float = 0.0):
        if img.ndim == 4 and img.shape[1] != self.cfg.in_channels:
            img = img.permute(0, 3, 1, 2)
        if prepared_text is None:
            txt, pooled_vec = self.prepare_text(context, attn_mask)
        else:
            txt, pooled_vec = prepared_text
        x = self.img_embedder(img)
        pos = get_2d_sincos_pos_embed(self.cfg.hidden_size, self.latent_img_size, x.device, x.dtype)
        x = x + pos[None]
        vec = self.t_embedder(t) + pooled_vec
        temporal_bias = None
        if temporal_frames > 1 and temporal_consistency > 0.0 and x.shape[0] % temporal_frames == 0:
            w_eff = min(max(float(temporal_consistency), 0.0), 1.0) * 0.8
            lt = txt.shape[1]
            li = x.shape[1]
            if temporal_frames == 2:
                temporal_bias = torch.empty(1, 1, 1, lt + 2 * li, device=x.device, dtype=x.dtype)
                temporal_bias[..., :lt] = 0.0
                temporal_bias[..., lt:lt + li] = math.log(1.0 - w_eff)
                temporal_bias[..., lt + li:] = math.log(w_eff)
            else:
                temporal_bias = torch.empty(1, 1, 1, lt + 3 * li, device=x.device, dtype=x.dtype)
                temporal_bias[..., :lt] = 0.0
                temporal_bias[..., lt:lt + li] = math.log(1.0 - w_eff)
                temporal_bias[..., lt + li:] = math.log(w_eff * 0.5)
        for block in self.double_blocks:
            x, txt = block(x, txt, vec, temporal_frames=temporal_frames, temporal_bias=temporal_bias)
        img_out = self.final_layer(x, vec)
        return self.unpatchify(img_out)


def tile_starts(length, size, overlap):
    if length <= size:
        return [0]
    n = math.ceil((length - size) / (size - overlap)) + 1
    return [(length - size) * i // (n - 1) for i in range(n)]


def _repeat_batch(tensor, k: int, b: int):
    if k == 1:
        return tensor
    if b == 1:
        return tensor.expand(k, *tensor.shape[1:])
    return tensor.repeat(k, *([1] * (tensor.ndim - 1)))


def _repeat_prepared(prepared, k: int, b: int):
    if prepared is None:
        return None
    return (_repeat_batch(prepared[0], k, b), _repeat_batch(prepared[1], k, b))


def _is_oom_error(exc: Exception) -> bool:
    oom_cls = getattr(torch, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_device_cache(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class DiffusionModel(nn.Module):
    def __init__(self, cfg: Optional[MMJiTConfig] = None):
        super().__init__()
        self.cfg = cfg or MMJiTConfig()
        self.net = MMJiT(self.cfg)

    def real_t_to_embed_t(self, t):
        return t

    def pred_velocity(self, x, t, text, mask):
        x0 = self.net(x, self.real_t_to_embed_t(t), text, mask)
        return (x0 - x) / torch.clamp(1 - t[:, None, None, None], min=0.001)

    def _cfg_velocity_and_x0(
        self,
        x,
        t,
        text,
        mask,
        cfg_scale: float,
        prepared_cond=None,
        prepared_uncond=None,
        temporal_frames: int = 1,
        temporal_consistency: float = 0.0,
    ):
        b = x.shape[0]
        use_cfg = (t >= self.cfg.cfg_interval[0]) & (t <= self.cfg.cfg_interval[1])
        if cfg_scale == 1.0 or not bool(use_cfg.any()):
            x0_cond = self.net(
                x, self.real_t_to_embed_t(t), text, mask,
                prepared_text=prepared_cond,
                temporal_frames=temporal_frames,
                temporal_consistency=temporal_consistency,
            )
            v = (x0_cond - x) / torch.clamp(1 - t[:, None, None, None], min=0.001)
            return v, x0_cond
        xx = torch.cat([x, x], dim=0)
        tt = torch.cat([t, t], dim=0)
        if prepared_cond is not None and prepared_uncond is not None:
            prepared_all = (
                torch.cat([prepared_cond[0], prepared_uncond[0]], dim=0),
                torch.cat([prepared_cond[1], prepared_uncond[1]], dim=0),
            )
            x0_all = self.net(
                xx, self.real_t_to_embed_t(tt), None, None,
                prepared_text=prepared_all,
                temporal_frames=temporal_frames,
                temporal_consistency=temporal_consistency,
            )
        else:
            yy = torch.cat([text, text], dim=0)
            mm = torch.cat([mask, torch.zeros_like(mask)], dim=0)
            x0_all = self.net(
                xx, self.real_t_to_embed_t(tt), yy, mm,
                temporal_frames=temporal_frames,
                temporal_consistency=temporal_consistency,
            )
        out = (x0_all - xx) / torch.clamp(1 - tt[:, None, None, None], min=0.001)
        cond, uncond = out[:b], out[b:]
        if bool(use_cfg.all()):
            v = uncond + (cond - uncond) * cfg_scale
        else:
            scale = torch.where(use_cfg[:, None, None, None], cfg_scale, 1.0).to(out.dtype)
            v = uncond + (cond - uncond) * scale
        return v, x0_all[:b]

    def cfg_velocity(self, x, t, text, mask, cfg_scale: float):
        v, _ = self._cfg_velocity_and_x0(x, t, text, mask, cfg_scale)
        return v

    def tile_weights(self, height, width, overlap, device, dtype):
        size = self.cfg.image_size
        ramp = torch.arange(size, device=device, dtype=torch.float32)
        ramp = torch.minimum(ramp + 1, size - ramp).clamp(max=overlap + 1) / (overlap + 1)
        window = ramp[:, None] * ramp[None, :]
        tiles = [(y, x) for y in tile_starts(height, size, overlap) for x in tile_starts(width, size, overlap)]
        total = torch.zeros(height, width, device=device, dtype=torch.float32)
        for y, x in tiles:
            total[y:y + size, x:x + size] += window
        return [(y, x, (window / total[y:y + size, x:x + size]).to(dtype)) for y, x in tiles]

    def tiled_cfg_velocity(
        self,
        x,
        t,
        text,
        mask,
        cfg_scale,
        tiles,
        tile_callback=None,
        return_x0=False,
        tile_batch_size: int = 1,
        prepared_cond=None,
        prepared_uncond=None,
        batch_state=None,
        temporal_consistency: float = 0.0,
    ):
        size = self.cfg.image_size
        b = x.shape[0]
        n_tiles = len(tiles)
        if n_tiles == 1 and x.shape[-2] == size and x.shape[-1] == size:
            v, x0_cond = self._cfg_velocity_and_x0(
                x, t, text, mask, cfg_scale,
                prepared_cond=prepared_cond,
                prepared_uncond=prepared_uncond,
                temporal_frames=b,
                temporal_consistency=temporal_consistency,
            )
            if tile_callback is not None:
                tile_callback(1)
            return (v, x0_cond) if return_x0 else v

        v = torch.zeros_like(x)
        x0_cond = torch.zeros_like(x) if return_x0 else None
        active_batch = batch_state[0] if batch_state is not None else max(1, int(tile_batch_size) // b)
        idx = 0
        while idx < n_tiles:
            remaining = n_tiles - idx
            n_chunks = math.ceil(remaining / active_batch)
            k = math.ceil(remaining / n_chunks)
            chunk = tiles[idx:idx + k]
            try:
                if k == 1:
                    y, x0, _ = chunk[0]
                    tile_x = x[:, :, y:y + size, x0:x0 + size]
                else:
                    tile_x = torch.cat([x[:, :, y:y + size, x0:x0 + size] for y, x0, _ in chunk], dim=0)
                t_c = _repeat_batch(t, k, b)
                if prepared_cond is not None:
                    p_cond = _repeat_prepared(prepared_cond, k, b)
                    p_uncond = _repeat_prepared(prepared_uncond, k, b)
                    text_c, mask_c = None, None
                else:
                    p_cond, p_uncond = None, None
                    text_c = _repeat_batch(text, k, b)
                    mask_c = _repeat_batch(mask, k, b)
                chunk_v, chunk_x0 = self._cfg_velocity_and_x0(
                    tile_x, t_c, text_c, mask_c, cfg_scale,
                    prepared_cond=p_cond,
                    prepared_uncond=p_uncond,
                    temporal_frames=b,
                    temporal_consistency=temporal_consistency,
                )
            except Exception as e:
                if active_batch <= 1 or not _is_oom_error(e):
                    raise
                _empty_device_cache(x.device)
                active_batch = max(1, k // 2)
                if batch_state is not None:
                    batch_state[0] = active_batch
                continue

            for ci, (y, x0, weight) in enumerate(chunk):
                sl = slice(ci * b, (ci + 1) * b)
                v[:, :, y:y + size, x0:x0 + size].addcmul_(chunk_v[sl], weight)
                if return_x0:
                    x0_cond[:, :, y:y + size, x0:x0 + size].addcmul_(chunk_x0[sl], weight)
            idx += k
            if tile_callback is not None:
                tile_callback(k)
        if return_x0:
            return v, x0_cond
        return v

    def tiled_cond_x0(self, x, t, text, mask, tiles, tile_batch_size: int = 1, prepared_cond=None, temporal_consistency: float = 0.0):
        size = self.cfg.image_size
        b = x.shape[0]
        n_tiles = len(tiles)
        if n_tiles == 1 and x.shape[-2] == size and x.shape[-1] == size:
            return self.net(x, t, text, mask, prepared_text=prepared_cond, temporal_frames=b, temporal_consistency=temporal_consistency)
        x0 = torch.zeros_like(x)
        active_batch = max(1, int(tile_batch_size) // b)
        idx = 0
        while idx < n_tiles:
            remaining = n_tiles - idx
            n_chunks = math.ceil(remaining / active_batch)
            k = math.ceil(remaining / n_chunks)
            chunk = tiles[idx:idx + k]
            tile_x = (
                x[:, :, chunk[0][0]:chunk[0][0] + size, chunk[0][1]:chunk[0][1] + size]
                if k == 1
                else torch.cat([x[:, :, y:y + size, x1:x1 + size] for y, x1, _ in chunk], dim=0)
            )
            t_c = _repeat_batch(t, k, b)
            p_cond = _repeat_prepared(prepared_cond, k, b)
            text_c = None if p_cond is not None else _repeat_batch(text, k, b)
            mask_c = None if p_cond is not None else _repeat_batch(mask, k, b)
            chunk_x0 = self.net(
                tile_x, t_c, text_c, mask_c,
                prepared_text=p_cond,
                temporal_frames=b,
                temporal_consistency=temporal_consistency,
            )
            for ci, (y, x1, weight) in enumerate(chunk):
                sl = slice(ci * b, (ci + 1) * b)
                x0[:, :, y:y + size, x1:x1 + size].addcmul_(chunk_x0[sl], weight)
            idx += k
        return x0

    def tiled_x0_vjp(self, params, x, t, text, mask, tiles, grad_x0, prepared_text=None, temporal_consistency: float = 0.0):
        # Vector-Jacobian product of the blended conditional x0 prediction w.r.t. x, one tile at a time
        # so activations are never held for the whole canvas.
        size = self.cfg.image_size
        b = x.shape[0]
        g = torch.zeros_like(grad_x0)
        t_c = t.clone()
        text_c = None if prepared_text is not None else text.clone()
        mask_c = None if prepared_text is not None else mask.clone()
        extra_kwargs = {"temporal_frames": b, "temporal_consistency": temporal_consistency}
        if prepared_text is not None:
            extra_kwargs["prepared_text"] = prepared_text
        for y, x0, weight in tiles:
            xt = x[:, :, y:y + size, x0:x0 + size].clone().requires_grad_(True)
            if prepared_text is not None:
                pred = functional_call(self.net, params, (xt, t_c, None, None), extra_kwargs)
            else:
                pred = functional_call(self.net, params, (xt, t_c, text_c, mask_c), extra_kwargs)
            tile_g, = torch.autograd.grad(pred, xt, grad_outputs=grad_x0[:, :, y:y + size, x0:x0 + size] * weight)
            g[:, :, y:y + size, x0:x0 + size] += tile_g
        return g

    @torch.no_grad()
    def sample(
        self,
        text,
        mask,
        cfg_scale=6.0,
        generator=None,
        progress=False,
        image=None,
        denoise=1.0,
        tile_overlap=64,
        tile_batch_size=1,
        temporal_consistency=0.0,
        shared_noise=0.0,
        flow_consistency=0.0,
        callback=None,
        guide=None,
    ):
        b = text.shape[0]
        device = text.device
        dtype = next(self.parameters()).dtype
        size = self.cfg.image_size
        height, width = image.shape[-2:] if image is not None else (size, size)
        shared = min(max(float(shared_noise), 0.0), 1.0)
        flow_strength = min(max(float(flow_consistency), 0.0), 1.0)
        bwd_flows = occ_masks = None
        if image is not None and b > 1 and flow_strength > 0.0:
            bwd_flows, occ_masks = compute_optical_flow(image)
        if b > 1 and shared > 0.0:
            if bwd_flows is not None:
                x = warp_noise_sequence(
                    b, self.cfg.in_channels, height, width,
                    bwd_flows, occ_masks, shared, device, dtype, generator=generator,
                ) * 2
            else:
                base_noise = torch.randn(1, self.cfg.in_channels, height, width, generator=generator, device=device, dtype=dtype)
                if shared >= 1.0:
                    x = base_noise.expand(b, -1, -1, -1).clone() * 2
                else:
                    frame_noise = torch.randn(b, self.cfg.in_channels, height, width, generator=generator, device=device, dtype=dtype)
                    x = (math.sqrt(shared) * base_noise + math.sqrt(1.0 - shared) * frame_noise) * 2
        else:
            x = torch.randn(b, self.cfg.in_channels, height, width, generator=generator, device=device, dtype=dtype) * 2
        t_start = 1.0 - denoise
        if image is not None:
            x = torch.lerp(x, image, t_start)
        tiles = self.tile_weights(height, width, tile_overlap, device, dtype)
        tile_callback = None
        if callback is not None:
            total = self.cfg.n_T * len(tiles)
            done = 0

            def tile_callback(n=1):
                nonlocal done
                done += n
                callback(done, total)
        text_dtype = text.to(dtype)
        mask_dtype = mask.to(dtype)
        cond_prepared = self.net.prepare_text(text_dtype, mask_dtype)
        uncond_prepared = self.net.prepare_text(text_dtype, torch.zeros_like(mask_dtype)) if cfg_scale != 1.0 else None
        if guide is not None:
            # ComfyUI loads weights under inference_mode and autograd can't save inference tensors,
            # so guidance backpropagates through a detached clone that lives only for this call.
            with torch.inference_mode(False):
                params = {k: p.detach().clone() for k, p in itertools.chain(self.net.named_parameters(), self.net.named_buffers())}
                vjp_prepared = (cond_prepared[0].detach().clone(), cond_prepared[1].detach().clone())
        batch_state = [max(1, int(tile_batch_size) // b)]
        timesteps = torch.linspace(t_start, 1.0, self.cfg.n_T + 1, device=device, dtype=dtype)
        iterator = range(self.cfg.n_T)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator)
        for i in iterator:
            t_cur = timesteps[i].expand(b)
            t_next = timesteps[i + 1].expand(b)
            t_float = t_start + (1.0 - t_start) * i / self.cfg.n_T
            need_guide = guide is not None and guide.active(t_float)
            if need_guide:
                v, x0_cond = self.tiled_cfg_velocity(
                    x, t_cur, text_dtype, mask_dtype, cfg_scale, tiles, tile_callback,
                    return_x0=True,
                    tile_batch_size=tile_batch_size,
                    prepared_cond=cond_prepared,
                    prepared_uncond=uncond_prepared,
                    batch_state=batch_state,
                    temporal_consistency=temporal_consistency,
                )
                remaining = torch.clamp(1 - t_cur, min=0.001)[:, None, None, None]
                vjp = lambda grad_x0: self.tiled_x0_vjp(
                    params, x, t_cur, text_dtype, mask_dtype, tiles, grad_x0,
                    prepared_text=vjp_prepared,
                    temporal_consistency=temporal_consistency,
                )
                v = v - guide.correction(x0_cond, t_float, vjp) / remaining
            else:
                v = self.tiled_cfg_velocity(
                    x, t_cur, text_dtype, mask_dtype, cfg_scale, tiles, tile_callback,
                    tile_batch_size=tile_batch_size,
                    prepared_cond=cond_prepared,
                    prepared_uncond=uncond_prepared,
                    batch_state=batch_state,
                    temporal_consistency=temporal_consistency,
                )
            if bwd_flows is not None:
                v = anchor_x0_velocity(x, v, t_cur, i, self.cfg.n_T, bwd_flows, occ_masks, flow_strength)
            x = x + (t_next - t_cur)[:, None, None, None] * v
        return x


import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from diffusers import DiffusionPipeline, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


transformers_logging.set_verbosity_error()


class MiniT2IFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        if train_t_schedule not in {"uniform", "lognorm"}:
            raise ValueError(f"Unsupported train_t_schedule: {train_t_schedule}")

    def sample_train_timesteps(self, batch_size, device, dtype=torch.float32, generator=None):
        if self.config.train_t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
        normal = normal * self.config.t_lognorm_sigma + self.config.t_lognorm_mu
        return torch.sigmoid(normal).to(dtype=dtype)

    def get_inference_timesteps(self, num_inference_steps=None, device=None, dtype=torch.float32):
        steps = int(num_inference_steps or self.config.num_inference_steps)
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)


class MiniT2IMMJiTModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        txt_input_size: int = 1024,
        hidden_size: int = 768,
        txt_hidden_size: int = 768,
        cond_vec_size: int = 768,
        depth_double: int = 17,
        txt_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6666666666666665,
        pca_channels: int = 128,
        prompt_length: int = 256,
        n_T: int = 100,
        prediction: str = "x",
        sampler: str = "euler",
        cfg_channels: int = 3,
        cfg_interval: tuple = (0.0, 1.0),
        llm: str = "google/flan-t5-large",
    ):
        super().__init__()
        cfg = MMJiTConfig(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            txt_input_size=txt_input_size,
            hidden_size=hidden_size,
            txt_hidden_size=txt_hidden_size,
            cond_vec_size=cond_vec_size,
            depth_double=depth_double,
            txt_preamble_depth=txt_preamble_depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            pca_channels=pca_channels,
            prompt_length=prompt_length,
            n_T=n_T,
            prediction=prediction,
            sampler=sampler,
            cfg_channels=cfg_channels,
            cfg_interval=tuple(cfg_interval),
            llm=llm,
        )
        self.model = DiffusionModel(cfg)

    @property
    def mmjit_config(self) -> MMJiTConfig:
        return self.model.cfg

    def forward(self, img, t, context, attn_mask):
        return self.model.net(img, t, context, attn_mask)

    def pred_velocity(self, x, t, text, mask):
        return self.model.pred_velocity(x, t, text, mask)

    def sample(self, text, mask, cfg_scale=6.0, generator=None, progress=False, image=None, denoise=1.0, tile_overlap=64, tile_batch_size=1, temporal_consistency=0.0, shared_noise=0.0, flow_consistency=0.0, callback=None, guide=None):
        return self.model.sample(
            text, mask, cfg_scale=cfg_scale, generator=generator, progress=progress,
            image=image, denoise=denoise, tile_overlap=tile_overlap, tile_batch_size=tile_batch_size,
            temporal_consistency=temporal_consistency, shared_noise=shared_noise,
            flow_consistency=flow_consistency,
            callback=callback, guide=guide,
        )


class MiniT2ITextToImagePipeline(nn.Module):
    def __init__(
        self,
        transformer: MiniT2IMMJiTModel,
        scheduler: Optional[MiniT2IFlowMatchScheduler] = None,
        tokenizer=None,
        text_encoder=None,
        text_encoder_name: str = "google/flan-t5-large",
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        if not isinstance(scheduler, MiniT2IFlowMatchScheduler):
            scheduler = MiniT2IFlowMatchScheduler(
                train_t_schedule=train_t_schedule,
                t_lognorm_mu=t_lognorm_mu,
                t_lognorm_sigma=t_lognorm_sigma,
                num_inference_steps=num_inference_steps,
            )
        self.transformer = transformer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.config = SimpleNamespace(
            text_encoder_name=text_encoder_name,
            train_t_schedule=scheduler.config.train_t_schedule,
            t_lognorm_mu=scheduler.config.t_lognorm_mu,
            t_lognorm_sigma=scheduler.config.t_lognorm_sigma,
            num_inference_steps=scheduler.config.num_inference_steps,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        torch_dtype: Optional[torch.dtype] = None,
        text_encoder_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        root = Path(pretrained_model_name_or_path)
        if not root.exists():
            root = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        transformer = MiniT2IMMJiTModel.from_pretrained(root / "transformer", torch_dtype=torch_dtype, **kwargs)
        scheduler_dir = root / "scheduler"
        if scheduler_dir.exists():
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
        else:
            scheduler = MiniT2IFlowMatchScheduler()
        text_encoder_name = transformer.mmjit_config.llm
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_name,
            torch_dtype=text_encoder_dtype,
            local_files_only=local_files_only,
        )
        return cls(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_name=text_encoder_name,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(save_directory / "transformer", **kwargs)
        self.scheduler.save_pretrained(save_directory / "scheduler")

    def _encode_prompt(self, prompt: Union[str, List[str]], device):
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
        if self.text_encoder is None:
            self.text_encoder = T5EncoderModel.from_pretrained(self.config.text_encoder_name)
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        cfg = self.transformer.mmjit_config
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.prompt_length,
        )
        input_ids = tokens.input_ids.to(device)
        attn = tokens.attention_mask.to(device)
        text = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        return text, attn

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
        image: Optional[torch.Tensor] = None,
        denoise: float = 1.0,
        tile_overlap: int = 64,
        tile_batch_size: int = 1,
        temporal_consistency: float = 0.0,
        shared_noise: float = 0.0,
        flow_consistency: float = 0.0,
        callback=None,
        guide=None,
        prompt_embeds: Optional[tuple] = None,
    ):
        device = next(self.transformer.parameters()).device
        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for p in prompt:
                prompt_batch.extend([p] * num_images_per_prompt)

        old_steps = self.transformer.mmjit_config.n_T
        self.transformer.model.cfg.n_T = int(num_inference_steps or self.scheduler.config.num_inference_steps)
        try:
            if prompt_embeds is not None:
                text, attn = prompt_embeds
                text = text.to(device)
                attn = attn.to(device)
            else:
                text, attn = self._encode_prompt(prompt_batch, device)
            model_dtype = next(self.transformer.parameters()).dtype
            images = self.transformer.sample(
                text.to(dtype=model_dtype),
                attn.to(dtype=model_dtype),
                cfg_scale=guidance_scale,
                generator=generator,
                progress=progress,
                image=image,
                denoise=denoise,
                tile_overlap=tile_overlap,
                tile_batch_size=tile_batch_size,
                temporal_consistency=temporal_consistency,
                shared_noise=shared_noise,
                flow_consistency=flow_consistency,
                callback=callback,
                guide=guide,
            )
        finally:
            self.transformer.model.cfg.n_T = old_steps

        if output_type == "pt":
            images = (images.clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).to(torch.float16).cpu()
            pass
        else:
            images = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            if output_type == "pil":
                images = [Image.fromarray(image) for image in images]
                
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


class MiniT2IPipeline(DiffusionPipeline):
    MODEL_ALIASES = {
        "b": "minit2i-b-16",
        "b16": "minit2i-b-16",
        "b-16": "minit2i-b-16",
        "base": "minit2i-b-16",
        "minit2i-b16": "minit2i-b-16",
        "minit2i-b-16": "minit2i-b-16",
        "minit2i-b/16": "minit2i-b-16",
        "l": "minit2i-l-16",
        "l16": "minit2i-l-16",
        "l-16": "minit2i-l-16",
        "large": "minit2i-l-16",
        "minit2i-l16": "minit2i-l-16",
        "minit2i-l-16": "minit2i-l-16",
        "minit2i-l/16": "minit2i-l-16",
    }

    def __init__(self):
        super().__init__()

    @classmethod
    def _resolve_model_type(cls, model_type: str) -> str:
        key = model_type.lower().replace("_", "-")
        if key not in cls.MODEL_ALIASES:
            choices = ", ".join(sorted(set(cls.MODEL_ALIASES)))
            raise ValueError(f"Unknown model_type={model_type!r}. Expected one of: {choices}")
        return cls.MODEL_ALIASES[key]

    @staticmethod
    def _resolve_root(
        repo_id_or_path: Union[str, os.PathLike],
        model_dir: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, os.PathLike]],
        local_files_only: bool,
    ) -> Path:
        root = Path(repo_id_or_path)
        if root.exists():
            return root
        allow_patterns = [
            f"{model_dir}/transformer/*",
            f"{model_dir}/scheduler/*",
        ]
        if not local_files_only:
            try:
                return Path(
                    snapshot_download(
                        repo_id=str(repo_id_or_path),
                        revision=revision,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        allow_patterns=allow_patterns,
                    )
                )
            except OSError:
                pass
        return Path(
            snapshot_download(
                repo_id=str(repo_id_or_path),
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=allow_patterns,
            )
        )

    @classmethod
    def load_transformer(cls,
        model_type: str = "b16",
        repo_id_or_path: Union[str, os.PathLike] = "MiniT2I/MiniT2I",
        revision: Optional[str] = None,
        local_files_only: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        model_dir = cls._resolve_model_type(model_type)
        root = cls._resolve_root(repo_id_or_path, model_dir, revision, cache_dir, local_files_only)
        model_root = root / model_dir
        transformer = MiniT2IMMJiTModel.from_pretrained(model_root / "transformer", torch_dtype=torch_dtype)
        transformer.eval()
        return transformer

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        transformer, text_encoder, tokenizer,
        model_dir: Path,
        **kwargs,
    ):
        scheduler_dir = model_dir / "scheduler"
        scheduler = _SCHEDULER_CACHE.get(scheduler_dir)
        if scheduler is None:
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
            _SCHEDULER_CACHE[scheduler_dir] = scheduler
        pipe = MiniT2ITextToImagePipeline(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_name=transformer.mmjit_config.llm,
        )
        return pipe(prompt=prompt, **kwargs)


def build_transformer_from_checkpoint(ckpt_path: Union[str, os.PathLike]) -> MiniT2IMMJiTModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = MMJiTConfig(**payload["config"])
    transformer = MiniT2IMMJiTModel(**asdict(cfg))
    prefixed = payload["state_dict"]
    state_dict = {}
    for key, value in prefixed.items():
        if key.startswith("net."):
            state_dict[f"model.{key}"] = value
        else:
            state_dict[f"model.{key}"] = value
    transformer.load_state_dict(state_dict, strict=True)
    return transformer
