
ComfyUI nodes for MiniT2I
https://huggingface.co/MiniT2I/MiniT2I

Workflows are in the templates section.


| Model | VRAM |
| -- | --|
| b16 | 2.6GB |
| l16 | 4.3GB |


Models, schedulers, and recent prompt embeddings are cached in memory (managed via ComfyUI's `ModelPatcher` for VRAM offloading), and attention uses PyTorch's fused `scaled_dot_product_attention`.

### Changes

* v1.0.2: Split up into loader & inference nodes so ComfyUI handles VRAM, quicker.
