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

_TEXT_ENCODER_CACHE = None
_TRANSFORMER_CACHE = {}
_DEPTH_MODEL_CACHE = {}
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
        global _TEXT_ENCODER_CACHE
        if _TEXT_ENCODER_CACHE is None:
            text_encoder = _load_pretrained_local_first(T5EncoderModel, "google/flan-t5-large", torch_dtype=torch.float32)
            text_encoder.eval()
            tokenizer = _load_pretrained_local_first(AutoTokenizer, "google/flan-t5-large")
            _TEXT_ENCODER_CACHE = ModelPatcher(
                MiniT2IModule(text_encoder, tokenizer),
                mm.text_encoder_device(),
                mm.text_encoder_offload_device(),
            )

        return io.NodeOutput(_TEXT_ENCODER_CACHE.clone(),)


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
        cached = _TRANSFORMER_CACHE.get(model_type)
        if cached is None:
            transformer = MiniT2IPipeline.load_transformer(model_type)
            cached = ModelPatcher(MiniT2IModule(transformer), mm.get_torch_device(), mm.unet_offload_device())
            _TRANSFORMER_CACHE[model_type] = cached
        return io.NodeOutput(cached.clone(),)


class MiniT2ISampler(io.ComfyNode):
    """
    An example node

    Class methods
    -------------
    define_schema (io.Schema):
        Tell the main program the metadata, input, output parameters of nodes.
    fingerprint_inputs:
        optional method to control when the node is re executed.
    check_lazy_status:
        optional method to control list of input names that need to be evaluated.

    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            Return a schema which contains all information about the node.
            Some types: "Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo".
            For outputs the "io.Model.Output" should be used, for inputs the "io.Model.Input" can be used.
            The type can be a "Combo" - this will be a list for selection.
        """
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
                    step=1, # Slider's step
                    display_mode=io.NumberDisplay.number,  # Cosmetic only: display as "number" or "slider"
                    lazy=True,  # Will only be evaluated if check_lazy_status requires it
                ),
                io.Float.Input(
                    "guidance",
                    default=2.5,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    round=0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    display_mode=io.NumberDisplay.number,
                    lazy=True,
                ),
                io.Model.Input("model", tooltip="Use MiniT2I Loader", optional=True),
                io.Model.Input("text_encoder", tooltip="Use MiniT2I Text Encoder Loader", optional=True),
                io.Int.Input(
                    "seed",
                    default=1,
                    step=1,
                    lazy=True,  # Will only be evaluated if check_lazy_status requires it
                ),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Only used with an image. Lower keeps more of the input image."),
                io.Image.Input("image", optional=True, tooltip="Input image for image to image. Images larger than 512x512 are sampled in overlapping 512x512 tiles at their own size."),
                io.Int.Input("tile_overlap", default=64, min=0, max=256, step=8, tooltip="Overlap in pixels between 512x512 tiles when the image is larger than 512x512."),
                io.Int.Input("tile_batch_size", default=4, min=1, max=64, step=1, tooltip="Maximum number of 512x512 tiles to evaluate in parallel per batch. Higher is faster on large images but uses more VRAM."),
                io.Float.Input("temporal_consistency", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Cross-frame self-attention strength across multi-frame batches (attends to anchor and previous frame). Reduces color/style flicker. 0 disables."),
                io.Float.Input("shared_noise", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Correlates initial noise across frames (0 = independent, 1 = shared). Automatically warps noise along optical flow when flow_consistency > 0 to avoid screen-door artifacts."),
                io.Float.Input("flow_consistency", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Optical-flow temporal locking for multi-frame image batches: warps shared noise along motion and anchors early-step x0 predictions across frames (fading out before final sharpening). 0 disables."),
                MiniT2IControlType.Input("control", optional=True, tooltip="Use MiniT2I Control"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

#    @classmethod
#    def check_lazy_status(cls, image, string_field, int_field, float_field, print_to_screen):
#        """
#            Return a list of input names that need to be evaluated.
#
#            This function will be called if there are any lazy inputs which have not yet been
#            evaluated. As long as you return at least one field which has not yet been evaluated
#            (and more exist), this function will be called again once the value of the requested
#            field is available.
#
#            Any evaluated inputs will be passed as arguments to this function. Any unevaluated
#            inputs will have the value None.
#        """
#        if print_to_screen == "enable":
#            return ["int_field", "float_field", "string_field"]
#        else:
#            return []



    @classmethod
    def execute(cls, prompt, steps, guidance, model, text_encoder, seed, denoise, tile_overlap, tile_batch_size=4, temporal_consistency=0.0, shared_noise=0.0, flow_consistency=0.0, image=None, control=None) -> io.NodeOutput:
        torch.manual_seed(seed)

        transformer = model.model.module
        te_module = text_encoder.model.module
        tokenizer = text_encoder.model.tokenizer
        prompt_length = transformer.mmjit_config.prompt_length
        cache_key = (id(te_module), prompt, prompt_length)
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

        num_images = 1
        if image is not None:
            size = transformer.mmjit_config.image_size
            image = image[..., :3].movedim(-1, 1)
            height, width = image.shape[-2:]
            if min(height, width) < size:
                scale = size / min(height, width)
                image = comfy.utils.common_upscale(image, max(size, round(width * scale)), max(size, round(height * scale)), "bilinear", "disabled")
            image = (image * 2 - 1).to(transformer.device, transformer.dtype)
            num_images = image.shape[0]

        if num_images > 1:
            text_emb = text_emb.expand(num_images, -1, -1)
            attn_mask = attn_mask.expand(num_images, -1)

        guide = None
        if control:
            size = transformer.mmjit_config.image_size
            height, width = image.shape[-2:] if image is not None else (size, size)
            guide = Guide(control, height, width, transformer.device)

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
            image=image,
            denoise=denoise,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            temporal_consistency=temporal_consistency,
            shared_noise=shared_noise,
            flow_consistency=flow_consistency,
            callback=callback,
            guide=guide,
            prompt_embeds=(text_emb, attn_mask),
            output_type="pt",
        )
 
        return io.NodeOutput(output.images,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def fingerprint_inputs(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


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
            depth_path = folder_paths.get_full_path_or_raise("geometry_estimation", DEPTH_ESTIMATOR)
            cached_depth = _DEPTH_MODEL_CACHE.get(depth_path)
            if cached_depth is None:
                cached_depth = comfy.sd.load_diffusion_model(depth_path)
                _DEPTH_MODEL_CACHE[depth_path] = cached_depth
            entry["depth_model"] = cached_depth.clone()
        return io.NodeOutput((control or []) + [entry],)
