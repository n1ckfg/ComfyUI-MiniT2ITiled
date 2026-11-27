# **Animating MiniT2I with Temporal Attention**

[https\://gemini.google.com/app/289c2d088da66908](https://gemini.google.com/app/289c2d088da66908)

*User prompt: Can minit2i be modified to work with temporally consistent animation like animatediff*

Response: **Yes, you can modify MiniT2I for temporally consistent animation using the AnimateDiff paradigm**, though you cannot plug in existing AnimateDiff checkpoints. Because MiniT2I is a pixel-space, flow-matching Transformer (**MM-JiT**) rather than a latent-space U-Net, you would need to build and train a custom temporal module for its architecture—or use a training-free attention hack for quick results.

### **Why Existing AnimateDiff Weights Won't Work**

> * **Architecture mismatch:** Standard AnimateDiff injects temporal layers into the convolutional/attention blocks of a 2D U-Net (like SD 1.5 or SDXL). MiniT2I uses a flat Transformer backbone (MM-JiT).  
> * **Pixel space vs. Latent space:** AnimateDiff operates on compressed VAE latents. MiniT2I has no VAE or image tokenizer; it generates directly on pixel patches (e.g., 16×16 patches) using a flow-matching objective.

### **How to Build an "AnimateDiff" for MiniT2I (Trained Approach)**

Because MiniT2I is small (258M parameters for MiniT2I-B/16) and has clean PyTorch/Diffusers and JAX codebases, it is actually a great candidate for academic-scale video adaptation:

> 1. **Inflate the Batch Dimension:** Treat a video clip of *T* frames as a batch of *B*×*T* independent images when passing through MiniT2I’s existing spatial MM-JiT blocks.  
> 2. **Insert Factorized Temporal Attention:** After each (or every few) MM-JiT blocks, reshape the tokens from (*B*⋅*T*,*N*,*C*) to (*B*⋅*N*,*T*,*C*) and apply a 1D temporal self-attention block with temporal positional embeddings (1D RoPE or sinusoidal).  
> 3. **Zero-Initialize Output Projections:** Initialize the output linear layer of every new temporal block to zero (the core AnimateDiff/ControlNet trick). At step 0, the model behaves identically to pretrained MiniT2I without corrupting its spatial weights.  
> 4. **Freeze & Train with Flow Matching:** Freeze all pretrained MiniT2I weights and train *only* the temporal modules on video-text pairs (e.g., OpenVid, WebVid, or Panda-70M) using the exact same flow-matching loss MiniT2I already uses.

### **Key Trade-offs of Pixel-Space Animation**

> * **No VAE flicker:** Without a 2D image VAE decoding each frame independently, you avoid classic latent-to-pixel decoding shimmer.  
> * **Higher token count:** Without a 3D video VAE compressing frames temporally, a 16-frame 512×512 clip at patch size 16 produces 16×1024=16,384 tokens. Using **factorized (1D) temporal attention** instead of full 3D spatio-temporal attention is essential to keep VRAM manageable (*O*(*N*⋅*T*2) instead of *O*((*N*⋅*T*)2)).  
> * **High-frequency jitter:** Pixel-space diffusion can exhibit subtle high-frequency texture jitter across frames. Using MiniT2I-B/16-MF (the 4-step Mean Flow distilled checkpoint) or adding a lightweight temporal smoothing / 3D conv layer in the motion module can help stabilize fine textures.

### **Faster Alternative: Training-Free Cross-Frame Attention**

If you don't want to train a motion module on video datasets, you can modify the MiniT2IPipeline in minit2i-torch for **zero-shot temporal consistency** (similar to Text2Video-Zero):

> * **Shared Keys/Values:** In the MM-JiT self-attention layers, replace each frame's Key (*K*) and Value (*V*) matrices with those of the first frame (*K*0​,*V*0​) or a blend of the first and previous frames (*K*0​,*Kt*−1​).  
> * **Noise Warping:** Apply a small affine shift or optical-flow warp to the initial Gaussian noise across frames so the flow-matching trajectory naturally pans or moves the scene.

This won't produce complex, natural physical motion like a trained motion module, but it takes only a few dozen lines of PyTorch to test.

**Sources**

> 1. [MiniT2I: A Minimalist Baseline for Text-to-Image Generation \| alphaXiv](https://www.alphaxiv.org/zh-TW/abs/2611.minit2i-a-minimalist-baseline-for-text-to-image-generation)

---

