"""A set of utility functions for working with pytorch tensors."""
import torch
import numpy as np
from itertools import product

from .torch_fft import fft, ifft


re = np.s_[..., 0]
im = np.s_[..., 1]


def iscomplex(a: torch.Tensor):
    """Return True if a is complex, False otherwise."""
    return a.shape[-1] == 2


def check_complex(A):
    """Raise a RuntimeWarning if tensor A is not complex."""
    for a in A:
        if not iscomplex(a):
            raise RuntimeWarning(
                "taking complex_mul of non-complex tensor! a.shape " + str(a.shape)
            )


def to_complex(real, imag=None):
    """Convert real and imaginary tensors to a complex tensor."""
    if imag is None:
        return torch.stack(
            [real, torch.zeros(real.size(), dtype=real.dtype, device=real.device)], -1
        )
    else:
        return torch.stack([real, imag], -1)


def get_device(device_type=None):
    """Initialize device cuda if available, CPU if no cuda is available."""
    if device_type is None and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    return device


def complex_matmul(a: torch.Tensor, b: torch.Tensor, conjugate=False) -> torch.Tensor:
    """
    Complex matrix multiplication of tensors a and b.

    Pass conjugate = True to conjugate tensor b in the multiplication.
    """
    check_complex([a, b])
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    if conjugate:
        real = are @ bre + aim @ bim
        imag = -are @ bim + aim @ bre
    else:
        real = are @ bre - aim @ bim
        imag = are @ bim + aim @ bre

    return torch.stack([real, imag], -1)


def complex_mul(a: torch.Tensor, b: torch.Tensor, conjugate=False) -> torch.Tensor:
    """
    Complex array multiplication of tensors a and b.

    Pass conjugate = True to conjugate tensor b in the multiplication.
    """
    check_complex([a, b])
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    if conjugate:
        real = are * bre + aim * bim
        imag = -are * bim + aim * bre
    else:
        real = are * bre - aim * bim
        imag = are * bim + aim * bre

    return torch.stack([real, imag], -1)


def torch_c_exp(angle):
    """Calculate exp(1j*angle)."""
    if angle.size()[-1] != 2:
        # Case of a real exponent
        result = torch.zeros(*angle.shape, 2, dtype=angle.dtype, device=angle.device)
        result[re] = torch.cos(angle)
        result[im] = torch.sin(angle)
    else:
        # Case of a complex valued exponent
        exp = torch.exp(-angle[im])
        result = torch.zeros(*angle.shape, dtype=angle.dtype, device=angle.device)
        result[re] = exp * torch.cos(angle[re])
        result[im] = exp * torch.sin(angle[re])
    return result


def sinc(x):
    """Calculate the sinc function ie. sin(pi x)/(pi x)."""
    y = torch.where(torch.abs(x) < 1.0e-20, torch.tensor([1.0e-20], dtype=x.dtype), x)
    return torch.sin(np.pi * y) / np.pi / y


def ensure_torch_array(array, dtype=torch.float, device=None):
    """
    Ensure that the input array is a pytorch tensor.

    Converts to a pytorch array if input is a numpy array and do nothing if the
    input is a pytorch tensor
    """
    from .. import (
        layered_structure_propagators,
        layered_structure_transmission_function,
    )

    if device is None:
        device = get_device(device)
    if isinstance(array, torch.Tensor):
        return array.to(device)
    elif isinstance(array, layered_structure_transmission_function):
        for i in range(len(array.Ts)):
            array.Ts[i] = array.Ts[i].to(device)
        return array
    elif isinstance(array, layered_structure_propagators):
        for i in range(len(array.Ps)):
            array.Ps[i] = array.Ps[i].to(device)
        return array
    else:
        if np.iscomplexobj(np.asarray(array)):
            return cx_from_numpy(np.asarray(array), dtype=dtype, device=device)
        else:
            return torch.from_numpy(np.asarray(array)).type(dtype).to(device)


def amplitude(r):
    """
    Calculate the amplitude of a complex tensor.

    If the tensor is not complex then calculate square.
    """
    if r.size(-1) == 2:
        return r[..., 0] * r[..., 0] + r[..., 1] * r[..., 1]
    else:
        return r * r


# def roll_n(X, axis, n):
#     """Roll a pytorch tensor X n entries along a given axis."""
#     f_idx = tuple(
#         slice(None, None, None) if i != axis % X.dim() else slice(0, n, None)
#         for i in range(X.dim())
#     )
#     b_idx = tuple(
#         slice(None, None, None) if i != axis % X.dim() else slice(n, None, None)
#         for i in range(X.dim())
#     )
#     front = X[f_idx]
#     back = X[b_idx]
#     return torch.cat([back, front], axis)


def cx_from_numpy(
    x: np.array, dtype=torch.float32, device=get_device()
) -> torch.Tensor:
    """
    Turn a complex numpy array into the required pytorch array format.

    Parameters
    ----------
    x : complex np.ndarray
        A complex numpy array

    Keyword arguments
    -----------------
    dtype : torch.dtype
        The datatype of the output array
    device : torch.device
        The device (CPU or GPU) of the output array
    """
    if "complex" in str(x.dtype):
        out = torch.zeros(*x.shape, 2)
        out[re] = torch.from_numpy(x.real)
        out[im] = torch.from_numpy(x.imag)
    else:
        if x.shape[-1] != 2:
            out = torch.zeros(x.shape + (2,))
            out[re] = torch.from_numpy(x.real)
        else:
            out = torch.zeros(x.shape + (2,))
            out[re] = torch.from_numpy(x[re])
            out[im] = torch.from_numpy(x[im])
    return out.to(device).type(dtype)


def cx_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a complex pytorch tensor to a complex numpy array."""
    check_complex(x)

    return x[re].cpu().numpy() + 1j * x[im].cpu().numpy()


def fftfreq(n, dtype=torch.float, device=torch.device("cpu")):
    """
    Generate an array of Fourier coordinates in units of pixels.

    Same as numpy.fft.fftfreq(n)*n but for a torch array.
    """
    return (torch.arange(n, dtype=dtype, device=device) + n // 2) % n - n // 2


def torch_dtype_to_numpy(dtype):
    """Convert a torch datatype to a numpy datatype."""
    scratch_array = torch.zeros(1, dtype=dtype)
    return scratch_array.cpu().numpy().dtype


def fourier_shift_array_1d(
    y, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """Apply Fourier shift theorem for sub-pixel shift to a 1 dimensional array."""
    ramp = torch.empty(y, 2, dtype=dtype, device=device)
    ky = 2 * np.pi * fftfreq(y) * posn
    if units == "pixels":
        ky /= y
    ramp[..., 0] = torch.cos(ky)
    ramp[..., 1] = -torch.sin(ky)
    return ramp


def fourier_shift_torch(
    array,
    posn,
    dtype=torch.float32,
    device=torch.device("cpu"),
    qspace_in=False,
    qspace_out=False,
    units="pixels",
):
    """
    Apply Fourier shift theorem for sub-pixel shifts to array.

    Parameters
    -----------
    array : torch.tensor (...,Y,X,2)
        Complex array to be Fourier shifted
    posn : torch.tensor (K x 2) or (2,)
        Shift(s) to be applied
    """
    if not qspace_in:
        array = fft(array, signal_ndim=2)

    array = complex_mul(
        array,
        fourier_shift_array(
            array.size()[-3:-1],
            posn,
            dtype=array.dtype,
            device=array.device,
            units=units,
        ),
    )

    if qspace_out:
        return array

    return ifft(array, signal_ndim=2)


def fourier_shift_array(
    size, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """
    Create Fourier shift theorem array to (pixel) position given by list posn.

    Parameters
    ----------
    size : array_like
        size of the array (Y,X)
    posn : array_like
        can be a K x 2 array to give a K x Y x X shift arrays
    posn
    """
    # Get number of dimensions
    nn = len(posn.shape)

    # Get size of array
    y, x = size

    if nn == 1:
        # Make y ramp exp(-2pi i ky y)
        yramp = fourier_shift_array_1d(
            y, posn[0], units=units, dtype=dtype, device=device
        )

        # Make y ramp exp(-2pi i kx x)
        xramp = fourier_shift_array_1d(
            x, posn[1], units=units, dtype=dtype, device=device
        )

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return complex_mul(yramp.view(y, 1, 2), xramp.view(1, x, 2))
    else:
        K = posn.shape[0]
        # Make y ramp exp(-2pi i ky y)
        yramp = torch.empty(K, y, 2, dtype=dtype, device=device)
        ky = (
            2
            * np.pi
            * fftfreq(y, dtype=dtype, device=device).view(1, y)
            * posn[:, 0].view(K, 1)
        )
        if units == "pixels":
            ky /= y
        yramp[..., 0] = torch.cos(ky)
        yramp[..., 1] = -torch.sin(ky)

        # Make y ramp exp(-2pi i kx x)
        xramp = torch.empty(K, x, 2, dtype=dtype, device=device)
        kx = (
            2
            * np.pi
            * fftfreq(x, dtype=dtype, device=device).view(1, x)
            * posn[:, 1].view(K, 1)
        )
        if units == "pixels":
            kx /= x

        xramp[..., 0] = torch.cos(kx)
        xramp[..., 1] = -torch.sin(kx)

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return complex_mul(yramp.view(K, y, 1, 2), xramp.view(K, 1, x, 2))


def crop_window_to_periodic_indices(win, shape):
    """
    Create indices for a rectangular subset of a larger array.

    If indices exceed the size of the larger array then these indices will wrap
    around to the other side of the grid providing two or more rectangular
    subsets of the larger array. Designed to be used in conjunction with
    the torch.narrow function to choose subsets of the square array to evaluate
    the PRISM algorithm on.

    Assumes that the requested window is smaller than the array size

    Parameters
    ----------
    win : (4,) array_like
        contains (y0,y,x0,x) the lower y index and y length and lower x index
        and x length
    shape : (2,) array_like
        Shape of the larger array

    Examples
    --------
    >>>> crop_window_to_periodic_indices([2,2,1,3],[5,5])
    (([2,2],[1,3]),)
    >>>> crop_window_to_periodic_indices([-1,3,1,3],[5,5])
    (([4,1],[1,3]),([0,2],[1,3]))
    >>>> crop_window_to_periodic_indices([4,4,1,3],[5,5])
    (([4,1],[1,3]),([0,3],[1,3]))
    >>>> list(crop_window_to_periodic_indices([4,4,3,3],[5,5]))
    (([4,1],[3,2]),([0,3],[3,2]),([4,1],[0,1]),([0,3],[0,1]))
    """

    def oneDindices(start, step, bound):
        if start + step > bound - 1:
            return [start, bound - start], [0, start + step - bound]
        elif start < 0:
            return [start % bound, bound - start % bound], [0, (start + step) % bound]
        else:
            return [[start, step]]

    y = oneDindices(*win[:2], shape[0])
    x = oneDindices(*win[2:], shape[1])

    return tuple(product(y, x))


def crop_window_to_flattened_indices_torch(indices: torch.Tensor, shape: list):
    """
    Create (flattened) indices for a rectangular subset of a larger array.

    Useful, for example for scattering matrix calculations where only a rectangular
    subset of the array is used in the PRISM interpolation routine

    Array indices exceeding the bounds of the array are wrapped to be consistent
    with periodic boundary conditions.

    Parameters
    ----------
    indices : torch.Tensor
        The centers of each of the cropping windows
    shape : array_like
        Size of the cropping windows

    Examples
    --------
    >>> indices = torch.as_tensor([[2,3,4],[1,2,3]])
    >>> gridshape = [4,4]
    >>> win = [3,3]
    >>> grid = torch.zeros(gridshape,dtype=torch.Long)
    tensor([[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
    >>> grid = grid.flatten()
    >>> ind = pyms.utils.crop_window_to_flattened_indices_torch(indices,gridshape)
    >>> grid[ind] = 1
    >>> grid.view(gridshape)
    tensor([[0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1]])
    """
    xind = torch.as_tensor(indices[-1]).view(1, len(indices[-1])) % shape[-1]
    yind = torch.as_tensor(indices[-2]).view(len(indices[-2]), 1) % shape[-2]
    return (xind + yind * shape[-1]).flatten().type(torch.LongTensor)


def crop_to_bandwidth_limit_torch(
    array: torch.Tensor,
    limit=2 / 3,
    qspace_in=True,
    qspace_out=True,
    norm="conserve_norm",
):
    """Crop an array to its bandwidth limit (remove superfluous array entries)."""
    # Check if array is complex or not
    complx = iscomplex(array)

    # Get array shape, taking into account final dimension of size 2 if the array
    # is complex
    gridshape = array.shape[-2 - int(complx) :][:2]

    # New shape of final dimensions
    newshape = tuple([int(round(gridshape[i] * limit)) for i in range(2)])

    return fourier_interpolate_2d_torch(
        array, newshape, norm=norm, qspace_in=qspace_in, qspace_out=qspace_out
    )


def size_of_bandwidth_limited_array(shape):
    """Get the size of an array after band-width limiting."""
    return list(crop_to_bandwidth_limit_torch(torch.zeros(*shape)).size())


def detect(detector, diffraction_pattern):
    """
    Apply a detector to a diffraction pattern.

    Calculates the signal in a diffraction pattern detector even if the size
    of the diffraction pattern and the detector are mismatched, assumes that
    the zeroth coordinate in reciprocal space is in the top-left hand corner
    of the array.
    """
    minsize = min(detector.size()[-2:], diffraction_pattern.size()[-2:])

    wind = [fftfreq(minsize[i], torch.long, detector.device) for i in [0, 1]]
    Dwind = crop_window_to_flattened_indices_torch(wind, detector.size())
    DPwind = crop_window_to_flattened_indices_torch(wind, diffraction_pattern.size())
    return torch.sum(
        detector.flatten(-2, -1)[:, None, Dwind]
        * diffraction_pattern.flatten(-2, -1)[None, :, DPwind],
        dim=-1,
    )


def fourier_interpolate_2d_torch(
    ain, shapeout, norm="conserve_val", qspace_in=False, qspace_out=False
):
    """
    Fourier interpolation of array ain to shape shapeout.

    If shapeout is smaller than ain.shape then Fourier downsampling is
    performed

    Parameters
    ----------
    ain : (...,Ny,Nx,2) torch.tensor
        Input array
    shapeout : (2,) array_like
        Shape of output array
    norm : str, optional  {'conserve_val','conserve_norm','conserve_L1'}
        Normalization of output. If 'conserve_val' then array values are preserved
        if 'conserve_norm' L2 norm is conserved under interpolation and if
        'conserve_L1' L1 norm is conserved under interpolation
    qspace_in : bool, optional
        If True expect a Fourier space input, otherwise (default) expect a
        real space input
    qspace_out : bool, optional
        If True return a Fourier space output, otherwise (default) return in
        real space
    """
    dtype = ain.dtype
    inputComplex = iscomplex(ain)
    # Make input complex
    aout = torch.zeros(
        ain.shape[: -2 - int(inputComplex)] + (np.prod(shapeout), 2),
        dtype=dtype,
        device=ain.device,
    )

    # Get input dimensions
    npiyin, npixin = ain.size()[-2 - int(inputComplex) :][:2]
    npiyout, npixout = shapeout

    # Get Fourier interpolation masks
    # PyTorch does not yet do element-wise logic operations, so we have to do
    # this bit in numpy. Additionally, in Windows pytorch does not support
    # bool types so we have to convert this to a unsigned 8-bit integer.
    from .numpy_utils import Fourier_interpolation_masks

    maskin, maskout = [
        torch.from_numpy(x).flatten()
        for x in Fourier_interpolation_masks(npiyin, npixin, npiyout, npixout)
    ]

    # Now transfer over Fourier coefficients from input to output array
    if inputComplex:
        ain_ = ain
    else:
        ain_ = to_complex(ain)

    if not qspace_in:
        ain_ = fft(ain_, signal_ndim=2)

    aout[..., maskout, :] = ain_.flatten(-3, -2)[..., maskin, :]

    # Fourier transform result with appropriate normalization
    if norm == "conserve_val":
        factor = npiyout * npixout / (npiyin * npixin)
    elif norm == "conserve_norm":
        factor = np.sqrt(npiyout * npixout / (npiyin * npixin))
    else:
        factor = 1

    # Fourier transform result with appropriate normalization
    aout = factor * aout.reshape(
        ain.shape[: -2 - int(inputComplex)] + tuple(shapeout) + (2,)
    )

    if not qspace_out:
        aout = ifft(aout, signal_ndim=2)

    # Return correct array data type
    if inputComplex:
        return aout
    return aout[re]


def crop_torch(arrayin, shapeout):
    """
    Crop the last two dimensions of arrayin to grid size shapeout.

    For entries of shapeout which are larger than the shape of the input array,
    perform zero-padding.
    """
    C = iscomplex(arrayin)

    # Number of dimensions in input array
    ndim = arrayin.ndim

    # Number of dimensions not covered by shapeout (ie not to be cropped)
    nUntouched = ndim - 2 - C

    # Shape of output array
    shapeout_ = arrayin.shape[:nUntouched] + tuple(shapeout)
    if C:
        shapeout_ += (2,)

    arrayout = torch.zeros(shapeout_, dtype=arrayin.dtype, device=arrayin.device)

    y, x = arrayin.shape[-2 - C :][:2]
    y_, x_ = shapeout[-2:]

    def indices(y, y_):
        if y > y_:
            # Crop in y dimension
            y1, y2 = [(y - y_) // 2, (y + y_) // 2]
            y1_, y2_ = [0, y_]
        else:
            # Zero pad in y dimension
            y1, y2 = [0, y]
            y1_, y2_ = [(y_ - y) // 2, (y + y_) // 2]
        return y1, y2, y1_, y2_

    y1, y2, y1_, y2_ = indices(y, y_)
    x1, x2, x1_, x2_ = indices(x, x_)

    if C:
        arrayout[..., y1_:y2_, x1_:x2_, :] = arrayin[..., y1:y2, x1:x2, :]
    else:
        arrayout[..., y1_:y2_, x1_:x2_] = arrayin[..., y1:y2, x1:x2]

    return arrayout
