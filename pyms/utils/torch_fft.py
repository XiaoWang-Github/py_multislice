import torch


def fft(x, *args, signal_ndim=None, **kwargs):
    try:
        from torch.fft import fft2

        if x.dtype in [torch.float16, torch.float32, torch.float64]:
            x = torch.view_as_complex(x)
        out = torch.view_as_real(fft2(x, *args, norm="ortho", **kwargs))
        return out
    except ImportError:
        from torch import fft

        return fft(x, *args, signal_ndim=signal_ndim, **kwargs)


def ifft(x, *args, signal_ndim=None, **kwargs):
    try:
        from torch.fft import ifft2

        if x.dtype in [torch.float16, torch.float32, torch.float64]:
            x = torch.view_as_complex(x)
        return torch.view_as_real(ifft2(x, *args, norm="ortho", **kwargs))
    except ImportError:
        from torch import ifft

        return ifft(x, *args, signal_ndim=signal_ndim, **kwargs)
