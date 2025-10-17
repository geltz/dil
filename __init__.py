import math
import torch
import torch.nn.functional as F

try:
    import comfy.model_management as mm
except Exception:
    # Fallbacks so the node can still run as a plain PyTorch script
    class _MM:
        @staticmethod
        def get_torch_device():
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        @staticmethod
        def should_use_fp16():
            # Heuristic: prefer fp32 for safety if unsure
            return False

        @staticmethod
        def should_use_bf16():
            return False
    mm = _MM()


def _pick_out_dtype():
    # Prefer fp32 for safety unless the runtime clearly prefers bf16/fp16.
    try:
        if hasattr(mm, "should_use_bf16") and mm.should_use_bf16():
            return torch.bfloat16
        if hasattr(mm, "should_use_fp16") and mm.should_use_fp16():
            return torch.float16
    except Exception:
        pass
    return torch.float32


def _safe_like(x, device=None, dtype=None):
    device = device if device is not None else (x.device if isinstance(x, torch.Tensor) else mm.get_torch_device())
    dtype = dtype if dtype is not None else (x.dtype if isinstance(x, torch.Tensor) else _pick_out_dtype())
    return dict(device=device, dtype=dtype)


def _unit_gauss(x, eps=1e-6):
    # Normalize to zero-mean, unit-std over B,C,H,W
    mean = x.mean(dim=(1,2,3), keepdim=True)
    std  = x.std(dim=(1,2,3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _sobel_kernels(device, dtype):
    # 3x3 Sobel
    gx = torch.tensor([[1., 0., -1.],
                       [2., 0., -2.],
                       [1., 0., -1.]], device=device, dtype=dtype) / 4.0
    gy = torch.tensor([[1.,  2.,  1.],
                       [0.,  0.,  0.],
                       [-1., -2., -1.]], device=device, dtype=dtype) / 4.0
    return gx, gy


def _gradients_depthwise(x):
    # x: [B,C,H,W] float32
    B, C, H, W = x.shape
    device = x.device
    dtype  = x.dtype
    gx, gy = _sobel_kernels(device, dtype)
    gx = gx.view(1,1,3,3).repeat(C,1,1,1)
    gy = gy.view(1,1,3,3).repeat(C,1,1,1)
    px = F.conv2d(x, gx, padding=1, groups=C)
    py = F.conv2d(x, gy, padding=1, groups=C)
    return px, py


def _gaussian_kernel2d(ks, sigma, device, dtype):
    # Build separable 2D Gaussian
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks-1)/2
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum().clamp_min(1e-12)
    g2 = g1[:, None] * g1[None, :]
    return g2


def _gaussian_blur_depthwise(x, sigma):
    # x: [B,C,H,W] float32
    B, C, H, W = x.shape
    # Kernel size ~ 6*sigma (odd)
    ks = int(max(3, math.ceil(sigma * 6.0)))
    if ks % 2 == 0:
        ks += 1
    k = _gaussian_kernel2d(ks, max(1e-6, float(sigma)), x.device, x.dtype)
    k = k.view(1,1,ks,ks).repeat(C,1,1,1)
    return F.conv2d(x, k, padding=ks//2, groups=C)


def _fft2(x):
    # fft over H,W for each channel; returns complex64 tensors (via float32 view)
    X = torch.fft.rfft2(x, norm="ortho")
    return X



def _radial_masks(H, W, device, dtype, splits=(0.15, 0.35)):
    """
    Build radial band masks for rfft2 output (H x (W//2+1)).
    Uses physical frequency bins to avoid shape mismatches.
    """
    # Frequencies for rows and rFFT columns
    fy = torch.fft.fftfreq(H, d=1.0).to(device=device, dtype=dtype)        # [H]
    fx = torch.fft.rfftfreq(W, d=1.0).to(device=device, dtype=dtype)       # [W//2+1]
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")                          # [H, Wf]
    r = torch.sqrt(yy * yy + xx * xx)
    r = r / r.max().clamp_min(1e-12)
    l_cut, h_cut = splits
    low  = r <= l_cut
    high = r >= h_cut
    mid  = (~low) & (~high)
    return low, mid, high


def _high_low_energy_fraction(x):
    # x: [B,C,H,W] float32
    B, C, H, W = x.shape
    X = _fft2(x)
    amp2 = (X.real**2 + X.imag**2)
    low, mid, high = _radial_masks(H, W, x.device, x.dtype, splits=(0.15, 0.35))
    # sum over H,W for each channel, then average over C and B
    def band(fr):
        return (amp2 * fr).sum(dim=(-2,-1))  # use mask for selection
    El = band(low).mean(dim=1).mean()    # average over C then B
    Eh = band(high).mean(dim=1).mean()
    Et = (amp2.sum(dim=(-2,-1)).mean(dim=1).mean()).clamp_min(1e-6)
    hf_frac = (Eh / Et).clamp(0.0, 1.0)
    lf_frac = (El / Et).clamp(0.0, 1.0)
    return hf_frac, lf_frac


def _orientation_coherence(px, py, eps=1e-6):
    # Structure-tensor coherence metric in [0,1], where 0 is isotropic.
    # Inputs: gradients [B,C,H,W] float32
    # Aggregate over channels and space
    Jxx = (px * px).mean(dim=(1,2,3))
    Jyy = (py * py).mean(dim=(1,2,3))
    Jxy = (px * py).mean(dim=(1,2,3))
    num = (Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2)
    den = (Jxx + Jyy).clamp_min(eps) ** 2
    coh = (num / den).clamp(0.0, 1.0)  # per-batch
    return coh.mean()                   # scalar


def _score(x):
    """
    Differentiable score to encourage:
      + edges without anisotropy
      + moderate high-frequency content
      + unit-Gaussian statistics
    """
    # Ensure float32 for stability
    x32 = x.to(torch.float32)
    # Edge term
    gx, gy = _gradients_depthwise(x32)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-9).mean()

    # Frequency balance (favor some high-frequency)
    hf, lf = _high_low_energy_fraction(x32)

    # Orientation penalty (penalize coherence/aniso)
    coh = _orientation_coherence(gx, gy)

    # Gaussian regularity
    mean = x32.mean()
    std  = x32.std().clamp_min(1e-6)
    gauss_reg = - (mean * mean + (std - 1.0) * (std - 1.0))

    # Final score (weights tuned conservatively)
    a, b, c, d = 1.0, 0.6, 0.1, 0.3
    S = a * edge + b * hf + c * gauss_reg - d * coh
    return S


def _precondition_dither(x, H, W):
    """
    Light "blue-ish" dither: unsharp mask to suppress coarse energy.
    Keeps isotropy by using isotropic Gaussian.
    """
    # sigma scaled by latent size (coarse blur)
    sigma = max(1.0, min(H, W) * 0.05)
    x_blur = _gaussian_blur_depthwise(x, sigma=sigma)
    # Unsharp-like boost, but very small to avoid artifacts
    x = x + 0.15 * (x - x_blur)
    return x


def _dil(latent_shape, seed, iters=2, eta=0.05, out_dtype=None, out_device=None):
    """
    Core DIL procedure.
    Returns a latent tensor shaped [B,4,H',W']
    """
    B, C, H, W = latent_shape
    device = out_device if out_device is not None else mm.get_torch_device()
    dtype  = out_dtype if out_dtype is not None else _pick_out_dtype()

    # Build a generator on the SAME device as the target tensor when possible.
    # Fall back to CPU generation + .to(device) to avoid device mismatch errors.
    if isinstance(device, torch.device):
        dev_type = device.type
    else:
        dev_type = str(device)

    if dev_type in ("cuda", "mps"):
        g = torch.Generator(device=device)
        gen_device = device
    else:
        g = torch.Generator(device="cpu")
        gen_device = torch.device("cpu")

    if seed is None or seed < 0:
        seed = int(torch.seed() % (2**31 - 1))
    g.manual_seed(int(seed))

    # Draw noise on gen_device to match the generator, then move if needed.
    x = torch.randn(B, C, H, W, device=gen_device, dtype=torch.float32, generator=g)
    if gen_device != device:
        x = x.to(device=device, non_blocking=True)

    # Light dither preconditioning
    x = _precondition_dither(x, H, W)
    # Normalize
    x = _unit_gauss(x)

    # Small number of gradient-ascent nudges with inference override
    with torch.inference_mode(False):   # not just enable_grad()
        for _ in range(int(max(0, iters))):
            x = x.detach().requires_grad_(True)
            S = _score(x)
            (grad,) = torch.autograd.grad(S, x, create_graph=False, retain_graph=False)
            x = (x + float(eta) * grad).detach()
            x = _unit_gauss(x)

    # Cast to requested output dtype at the very end
    x = x.to(device=device, dtype=dtype, non_blocking=True)
    return x

class DIL_EmptyLatent:
    """
    Dithered Isotropic Latent
    Training-free "smart noise" for diffusion

    Visible UI: width, height, batch_size
    Advanced (hidden): seed, iters, eta, channels
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                # Advanced controls with safe defaults
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31-1, "step": 1, "display": "number", "hidden": True}),
                "iters": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1, "display": "slider", "hidden": True}),
                "eta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.005, "display": "slider", "hidden": True}),
                "channels": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "hidden": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "make"
    CATEGORY = "latent"

    @staticmethod
    def _latent_wh(width, height):
        # Clamp to multiples of 8 for SD/SDXL VAEs
        w = max(8, int(width)) // 8
        h = max(8, int(height)) // 8
        return w, h

    def make(self, width, height, batch_size=1, seed=-1, iters=2, eta=0.05, channels=4):
        # Resolve device/dtype
        device = mm.get_torch_device()
        out_dtype = _pick_out_dtype()

        Wl, Hl = self._latent_wh(width, height)
        shape = (int(batch_size), int(channels), Hl, Wl)

        with torch.inference_mode(False):
            x = _dil(shape, seed=seed, iters=iters, eta=eta, out_dtype=out_dtype, out_device=device)

        latent = {"samples": x}
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "DIL_EmptyLatent": DIL_EmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DIL_EmptyLatent": "Dithered Isotropic Latent",
}
