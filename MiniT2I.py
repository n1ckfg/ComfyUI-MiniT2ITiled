from collections import OrderedDict
from comfy_api.latest import io
import torch
from .pipeline import MiniT2IPipeline
from .control import Guide, CONTROL_TYPES
from comfy.model_patcher import ModelPatcher
import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths
from transformers import AutoTokenizer, T5EncoderModel


MiniT2IControlType = io.Custom("MINIT2I_CONTROL")
# Estimates depth of the image being generated so depth and normal controls can compare it with the control map.
DEPTH_ESTIMATOR = "depth_anything_3_small.safetensors"

_PROMPT_CACHE = OrderedDict()
_PROMPT_CACHE_MAX = 16
_PIPELINE = MiniT2IPipeline()


def _load_pretrained_local_first(loader_cls, repo_id, **kwargs):
    try:
        return loader_cls.from_pretrained(repo_id, local_files_only=True, **kwargs)
    except OSError:
        return loader_cls.from_pretrained(repo_id, **kwargs)


class MiniT2IModule(torch.nn.Module):
    # ModelPatcher assigns .device, which is a read-only property on transformers/diffusers models
    def __init__(self, module, tokenizer=None):
        super().__init__()
        self.module = module
        self.tokenizer = tokenizer


class MiniT2ITextEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniT2ITextEncoder",
            display_name="MiniT2I Text Encoder Loader",
            category="MiniT2I",
            inputs=[
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        text_encoder = _load_pretrained_local_first(T5EncoderModel, "google/flan-t5-large", torch_dtype=torch.float32)
        text_encoder.eval()
        tokenizer = _load_pretrained_local_first(AutoTokenizer, "google/flan-t5-large")
        patcher = ModelPatcher(MiniT2IModule(text_encoder, tokenizer), mm.text_encoder_device(), mm.text_encoder_offload_device())
        return io.NodeOutput(patcher,)


class MiniT2ILoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniT2ILoader",
            display_name="MiniT2I Model Loader",
            category="MiniT2I",
            inputs=[
                io.Combo.Input("model_type", options=["b16","l16"], tooltip="l16=large, b16=normal"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_type) -> io.NodeOutput:
        transformer = MiniT2IPipeline.load_transformer(model_type)
        return io.NodeOutput(ModelPatcher(MiniT2IModule(transformer), mm.get_torch_device(), mm.unet_offload_device()),)


class MiniT2ISampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniT2I",
            display_name="MiniT2I Sampler",
            category="MiniT2I",
            inputs=[
                io.String.Input("prompt", multiline=True, lazy=True),
                io.Int.Input(
                    "steps",
                    default=10,
                    min=0,
                    max=4096,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    lazy=True,
                ),
                io.Float.Input(
                    "guidance",
                    default=2.5,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    round=0.001,
                    display_mode=io.NumberDisplay.number,
                    lazy=True,
                ),
                io.Model.Input("model", tooltip="Use MiniT2I Loader", optional=True),
                io.Model.Input("text_encoder", tooltip="Use MiniT2I Text Encoder Loader", optional=True),
                io.Int.Input(
                    "seed",
                    default=1,
                    step=1,
                    lazy=True,
                ),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Only used with an image. Lower keeps more of the input image."),
                io.Image.Input("image", optional=True, tooltip="Input image for image to image. Images larger than 512x512 are sampled in overlapping 512x512 tiles at their own size."),
                io.Int.Input("tile_overlap", default=64, min=0, max=256, step=8, tooltip="Overlap in pixels between 512x512 tiles when the image is larger than 512x512."),
                io.Int.Input("tile_batch_size", default=4, min=1, max=64, step=1, tooltip="Maximum number of 512x512 tiles to evaluate in parallel per batch. Higher is faster on large images but uses more VRAM."),
                io.Float.Input("temporal_consistency", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Cross-frame self-attention strength across multi-frame batches (attends to anchor and previous frame). Reduces color/style flicker. 0 disables."),
                io.Float.Input("shared_noise", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Correlates initial noise across frames (0 = independent, 1 = shared). When flow_consistency > 0 on an image batch, noise is warped along optical flow at max(shared_noise, flow_consistency)."),
                io.Float.Input("flow_consistency", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Optical-flow temporal locking for multi-frame image batches: warps noise along motion, guides cross-frame attention, and anchors early-step x0 predictions across frames (fading out before final sharpening). 0 disables."),
                io.Int.Input("width", default=512, min=512, max=16384, step=8, tooltip="Canvas width when no image is connected (defaults to the first control map's size if left at 512x512). Larger canvases are sampled in overlapping 512x512 tiles."),
                io.Int.Input("height", default=512, min=512, max=16384, step=8, tooltip="Canvas height when no image is connected (defaults to the first control map's size if left at 512x512)."),
                io.Float.Input("global_structure", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Canvases at least 1024 on the short side: follows the composition of a downscaled whole-canvas pass (fading out late in sampling) so the 512x512 tiles don't each compose their own scene. Use about 1 for text to image or high denoise. 0 disables."),
                MiniT2IControlType.Input("control", optional=True, tooltip="Use MiniT2I Control"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, prompt, steps, guidance, model, text_encoder, seed, denoise, tile_overlap, tile_batch_size=4, temporal_consistency=0.0, shared_noise=0.0, flow_consistency=0.0, width=512, height=512, global_structure=0.0, image=None, control=None) -> io.NodeOutput:
        transformer = model.model.module
        te_module = text_encoder.model.module
        tokenizer = text_encoder.model.tokenizer
        prompt_length = transformer.mmjit_config.prompt_length
        cache_key = (prompt, prompt_length)
        cached_embeds = _PROMPT_CACHE.get(cache_key)

        depth_models = [c["depth_model"] for c in control or [] if "depth_model" in c]
        if cached_embeds is not None:
            _PROMPT_CACHE.move_to_end(cache_key)
            mm.load_models_gpu([model] + depth_models)
            text_emb, attn_mask = cached_embeds
        else:
            mm.load_models_gpu([model, text_encoder] + depth_models)
            te_device = next(te_module.parameters()).device
            with torch.no_grad():
                tokens = tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=prompt_length,
                )
                input_ids = tokens.input_ids.to(te_device)
                attn_mask = tokens.attention_mask.to(te_device)
                text_emb = te_module(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            _PROMPT_CACHE[cache_key] = (text_emb.detach().cpu(), attn_mask.detach().cpu())
            if len(_PROMPT_CACHE) > _PROMPT_CACHE_MAX:
                _PROMPT_CACHE.popitem(last=False)

        size = transformer.mmjit_config.image_size
        batch_sizes = ([image.shape[0]] if image is not None else []) + [c["image"].shape[0] for c in (control or [])]
        num_images = max(batch_sizes, default=1)
        for bs in batch_sizes:
            if bs not in (1, num_images):
                raise ValueError(f"Incompatible batch sizes across image and control inputs: {batch_sizes}")

        if image is not None:
            image = image[..., :3].movedim(-1, 1)
            height, width = image.shape[-2:]
            if min(height, width) < size:
                scale = size / min(height, width)
                image = comfy.utils.common_upscale(image, max(size, round(width * scale)), max(size, round(height * scale)), "bilinear", "disabled")
                height, width = image.shape[-2:]
            image = (image * 2 - 1).to(transformer.device, torch.float32)
        elif control and (width, height) == (size, size):
            height, width = control[0]["image"].shape[1:3]
            if min(height, width) < size:
                scale = size / min(height, width)
                width, height = max(size, round(width * scale)), max(size, round(height * scale))

        if num_images > 1:
            text_emb = text_emb.expand(num_images, -1, -1)
            attn_mask = attn_mask.expand(num_images, -1)

        guide = Guide(control, height, width, transformer.device) if control else None

        pbar = comfy.utils.ProgressBar(steps)

        def callback(done, total):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(done, total)

        output = _PIPELINE(
            prompt,
            transformer, te_module, tokenizer,
            model_dir=transformer.config._name_or_path.parent,
            guidance_scale=guidance,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            generator=torch.Generator().manual_seed(seed),
            image=image,
            height=height,
            width=width,
            denoise=denoise,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            temporal_consistency=temporal_consistency,
            shared_noise=shared_noise,
            flow_consistency=flow_consistency,
            global_structure=global_structure,
            callback=callback,
            guide=guide,
            prompt_embeds=(text_emb, attn_mask),
            output_type="pt",
        )

        return io.NodeOutput(output.images.to(mm.intermediate_device(), mm.intermediate_dtype()),)


class MiniT2IControl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniT2IControl",
            display_name="MiniT2I Control",
            category="MiniT2I",
            inputs=[
                io.Image.Input("image", tooltip="Control map. edge: white lines on black (Canny, Lineart, HED). depth: near = white. normal: ControlNet/BAE convention."),
                io.Combo.Input("control_type", options=list(CONTROL_TYPES)),
                io.Float.Input("strength", default=0.4, min=0.0, max=4.0, step=0.05),
                io.Float.Input("start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Sampling progress where guidance starts."),
                io.Float.Input("end", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Sampling progress where guidance stops. Later steps only refine detail."),
                MiniT2IControlType.Input("control", optional=True, tooltip="Previous MiniT2I Control, to combine several."),
            ],
            outputs=[
                MiniT2IControlType.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, control_type, strength, start, end, control=None) -> io.NodeOutput:
        entry = {"type": control_type, "image": image, "strength": strength, "start": start, "end": end}
        if control_type != "edge":
            entry["depth_model"] = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("geometry_estimation", DEPTH_ESTIMATOR))
        return io.NodeOutput((control or []) + [entry],)
