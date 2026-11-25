from comfy_api.latest import io
import torch
from .pipeline import MiniT2IPipeline
from comfy.model_patcher import ModelPatcher
import comfy.model_management as mm
import comfy.utils
from transformers import AutoTokenizer, T5EncoderModel


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
                io.Image.Input("image", optional=True, tooltip="Input image for image to image. Resized to the model's 512x512."),
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
    def execute(cls, prompt, steps, guidance, model, text_encoder, seed, denoise, image=None) -> io.NodeOutput:
        torch.manual_seed(seed)

        mm.load_models_gpu([model, text_encoder])
        transformer = model.model.module
        num_images = 1
        if image is not None:
            size = transformer.mmjit_config.image_size
            image = image[..., :3].movedim(-1, 1)
            if image.shape[-2:] != (size, size):
                image = comfy.utils.common_upscale(image, size, size, "bilinear", "center")
            image = (image * 2 - 1).to(transformer.device, transformer.dtype)
            num_images = image.shape[0]
        output = MiniT2IPipeline()(
            prompt,
            transformer, text_encoder.model.module, text_encoder.model.tokenizer,
            model_dir=transformer.config._name_or_path.parent,
            guidance_scale=guidance,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            image=image,
            denoise=denoise,
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

