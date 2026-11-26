from comfy_api.latest import io
import torch
from .pipeline import MiniT2IPipeline
from .control import Guide, CONTROL_TYPES
from comfy.model_patcher import ModelPatcher
import comfy.model_management as mm
import comfy.utils
from transformers import AutoTokenizer, T5EncoderModel


MiniT2IControlType = io.Custom("MINIT2I_CONTROL")
DA3ModelType = io.Custom("DA3_MODEL")


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
        text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        return io.NodeOutput(ModelPatcher(MiniT2IModule(text_encoder, tokenizer), mm.text_encoder_device(), mm.text_encoder_offload_device()),)


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
    def execute(cls, prompt, steps, guidance, model, text_encoder, seed, denoise, tile_overlap, image=None, control=None) -> io.NodeOutput:
        torch.manual_seed(seed)

        mm.load_models_gpu([model, text_encoder] + [c["depth_model"] for c in control or [] if "depth_model" in c])
        transformer = model.model.module
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

        guide = None
        if control:
            size = transformer.mmjit_config.image_size
            height, width = image.shape[-2:] if image is not None else (size, size)
            guide = Guide(control, height, width, transformer.device)

        pbar = comfy.utils.ProgressBar(steps)

        def callback(done, total):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(done, total)

        output = MiniT2IPipeline()(
            prompt,
            transformer, text_encoder.model.module, text_encoder.model.tokenizer,
            model_dir=transformer.config._name_or_path.parent,
            guidance_scale=guidance,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            image=image,
            denoise=denoise,
            tile_overlap=tile_overlap,
            callback=callback,
            guide=guide,
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
                io.Image.Input("image", tooltip="Control map, e.g. from a Canny, Lineart or HED preprocessor."),
                io.Combo.Input("control_type", options=list(CONTROL_TYPES)),
                io.Float.Input("strength", default=0.4, min=0.0, max=4.0, step=0.05),
                io.Float.Input("start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Sampling progress where guidance starts."),
                io.Float.Input("end", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Sampling progress where guidance stops. Later steps only refine detail."),
                MiniT2IControlType.Input("control", optional=True, tooltip="Previous MiniT2I Control, to combine several."),
                DA3ModelType.Input("depth_model", optional=True, tooltip="Load Depth Anything 3 (Small is fastest). Required for depth and normal."),
            ],
            outputs=[
                MiniT2IControlType.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, control_type, strength, start, end, control=None, depth_model=None) -> io.NodeOutput:
        entry = {"type": control_type, "image": image, "strength": strength, "start": start, "end": end}
        if control_type != "edge":
            if depth_model is None:
                raise ValueError(f"MiniT2I Control: '{control_type}' needs a depth_model from Load Depth Anything 3.")
            entry["depth_model"] = depth_model
        return io.NodeOutput((control or []) + [entry],)
