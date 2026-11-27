
ComfyUI nodes for MiniT2I
https://huggingface.co/MiniT2I/MiniT2I

Workflows are in the templates section.


| Model | VRAM |
| -- | --|
| b16 | 2.6GB |
| l16 | 4.3GB |


Models are managed by ComfyUI's `ModelPatcher` for VRAM offloading, recent prompt embeddings are cached in memory, and attention uses PyTorch's fused `scaled_dot_product_attention`.

For text to image at 1080p and above, set `width`/`height` and use `global_structure` ≈ 1 so the 512×512 tiles share one composition.

### Changes

* v1.0.2: Split up into loader & inference nodes so ComfyUI handles VRAM, quicker.
