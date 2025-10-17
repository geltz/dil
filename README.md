# Dithered Isotropic Latent

Generate **training-free “smart noise”** latents that are isotropic, lightly textured, and numerically well-conditioned for diffusion. The node optimizes a tiny objective a few steps over Gaussian noise, then re-normalizes—yielding latents that tend to start samplers in a healthier part of the space than plain i.i.d. noise.  

## Highlights

* **Drop-in ComfyUI node**: `DIL_EmptyLatent` (“Dithered Isotropic Latent”) appears under the *latent* category. 
* **Clean UI**: only **width**, **height**, **batch_size** are visible; **seed / iters / eta / channels** live under *Advanced*.  
* **Sampler-friendly shape**: outputs a standard diffusion latent `[B, 4, H', W']`, with `H' = H/8`, `W' = W/8`, and inputs clamped to multiples of 8 for SD/SDXL VAEs.  
* **Safe defaults**: `iters=2`, `eta=0.05`; robust device/dtype handling and graceful fallbacks when Comfy internals aren’t present.  

## Installation

1. **Clone or copy** this repository into your ComfyUI custom nodes directory:

   ```
   ComfyUI/custom_nodes/dil-empty-latent
   ```
2. **Restart ComfyUI**. You should see **“Dithered Isotropic Latent”** in the *latent* category. 

> DIL also runs as a plain PyTorch script thanks to internal fallbacks for device and dtype management (no Comfy dependency required at runtime). 

## Quickstart (ComfyUI)

1. Drop **Dithered Isotropic Latent** into your graph.
2. Set **width / height / batch_size** (only these are visible).
3. (Optional) Expand **Advanced** to set **seed**, **iters**, **eta**, **channels**.
4. Wire the **latent** output to your sampler.

All UI fields and defaults are defined in the node class: `width/height/batch_size` (visible) and `seed/iters/eta/channels` (hidden). 

## How it works

**Start with Gaussian noise + tiny blue-ish dither, then optimize a small score for a few steps (with renormalization each time):**

1. **Precondition**
   Draw `x ~ N(0, I)` in `[B,C,H,W]`, apply a very light unsharp-mask style dither
   `x ← x + 0.15 * (x − gaussian_blur(x, σ ≈ 0.05 * min(H,W)))`, then unit-normalize per sample/channel.  

2. **Score to maximize**

* Edge strength via Sobel: `edge = mean(sqrt(gx^2 + gy^2 + 1e−9))`
* Frequency balance using rFFT radial masks (low ≤ 0.15, high ≥ 0.35): high-freq fraction `hf = Eh/Et`
* Orientation coherence penalty from the structure tensor:
  `coh = ((Jxx−Jyy)^2 + 4*Jxy^2) / (Jxx+Jyy)^2` (clamped to [0,1])
* Gaussian regularity: `gauss_reg = −(mean(x)^2 + (std(x)−1)^2)`
* Final scalar objective:

  ```
  S(x) = 1.0*edge + 0.6*hf + 0.1*gauss_reg − 0.3*coh
  ```

3. **Few gradient-ascent nudges**
   Repeat `iters` times: `x ← x + η * ∇x S(x)`; re-normalize to unit Gaussian after each step. Implemented under `torch.inference_mode(False)` to allow gradients inside the loop. Defaults: `iters=2`, `eta=0.05`.   

4. **Output**
   Cast once to the target device/dtype and return a diffusion-ready latent `[B, 4, H', W']`. Shapes are clamped to multiples of 8 for SD/SDXL VAEs.  

## Node API

* **Class / Mapping**: `DIL_EmptyLatent` (exported via `NODE_CLASS_MAPPINGS`) with display name **“Dithered Isotropic Latent.”** 
* **Category**: `latent` • **Function**: `make` • **Return**: `("LATENT",)` named `("latent",)`. 
* **Inputs**

  * **Required**: `width`, `height`, `batch_size`
  * **Advanced (hidden)**: `seed`, `iters`, `eta`, `channels` (defaults shown in code) 
* **Dimension handling**: `width`/`height` are converted to latent dims by dividing by 8 (integer) with clamping. 

## Implementation notes

* **Device & dtype safety**

  * Picks output dtype based on Comfy preferences; defaults to `float32` if uncertain. 
  * Falls back to a local device manager if Comfy’s `model_management` isn’t available (still runs as a plain script). 
* **Generator locality & seeding**

  * Builds a `torch.Generator` on the *same device* when possible; otherwise draws on CPU and transfers. Seeds are set robustly; negative/None seeds auto-randomize.  
* **Single cast at the end**

  * Computation stays in stable precision; final cast/device move happens once. 

## Tips

* Start with the defaults (`iters=2`, `eta=0.05`). Increase `iters` sparingly if you want slightly stronger edge/high-freq encouragement without breaking isotropy.  
* Keep `channels=4` for SD/SDXL unless you know you need something custom. 

## Development

* Core gradient loop and re-normalization happen under `torch.inference_mode(False)` to allow autograd, with a per-step unit-Gaussian clamp for stability. 
* The score combines **edges**, **frequency balance**, **Gaussian regularity**, and **orientation coherence** with conservative weights `a=1.0, b=0.6, c=0.1, d=0.3`. 