# Variable/function aliases for code conciseness (avoid module boilerplates)
# Plus other useful functions

#### Numpy stuff ####
import numpy as np

# Numpy constants
pi = np.pi
e = np.e

# Numpy array allocation
array = np.array
arange = np.arange
zeros = np.zeros
ones = np.ones
empty = np.empty

# Numpy basic array concatenation
hstack = np.hstack
vstack = np.vstack

# Other numpy array operations
flip = np.flip
roll = np.roll

# Numpy random sample generation
seed = np.random.seed
randn = np.random.randn
rand = np.random.rand
randi = randint = np.random.randint  # With a shortening alias

# Numpy basic mathematical functions
sqrt = np.sqrt
exp = np.exp
sin = np.sin
cos = np.cos
sinc = np.sinc
floor = np.floor
ceil = np.ceil
min = np.min
max = np.max
round = np.round
abs = np.abs
sum = np.sum
angle = np.angle
real = np.real
imag = np.imag
conj = conjugate = np.conjugate  # With a shortening alias

# Numpy convolution and FFT
conv = convolve = np.convolve  # With a shortening alias
fft = np.fft.fft
fftshift = np.fft.fftshift


# Extra custom functions

def iarange(start, stop, step=1):
    """Exactly like `arange`, but also includes the last point."""
    return arange(start, stop + step/2, step)

def upsample(x: np.ndarray, L: int):
    """Upsamples a numpy 1D signal `x` by an integer factor `L`."""
    y = zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y


def downsample(x, L: int):
    """Downsamples a 1D signal `x` by an integer factor `L`."""
    return x[::L]


def convxaxis(x_axis, h_axis):
    ''' input:
    x_axis: 1D array, time axis of x[n]
    h_axis: 1D array, time axis of h[n]

        returns:
    y_axis: 1D array, time axis for the convolution result.
    '''
    # Check if x_axis and h_axis have at least two points
    if len(x_axis) < 2 or len(h_axis) < 2:
        raise ValueError("x_axis and h_axis must each have at least two points.")

    # sampling period of each signal is the time diff between two consecutive points
    Ts_x = x_axis[1] - x_axis[0]
    Ts_h = h_axis[1] - h_axis[0]

    # Check if sampling periods match
    if abs(Ts_x - Ts_h) > 1e-12:
        raise ValueError("Sampling periods must match.")

    # calculate sampling period and number of points for the convolution result
    Ts = Ts_x
    N_y = len(x_axis) + len(h_axis) - 1
    start = x_axis[0] + h_axis[0]

    # return the time axis for the convolution result
    return start + arange(N_y) * Ts


def fftxaxis(N: int, Ts: float):
    '''
    Returns the frequency axis for an N-point FFT with sampling period `Ts`.
    '''
    # Calculate the sampling frequency
    fs = 1 / Ts
    # Return the frequency axis
    return arange(-fs/2, fs/2, fs/N)



del np


#### matplotlib.pyplot stuff ####
import matplotlib.pyplot as plt

# Figure and plotting
figure = plt.figure
subplot = plt.subplot
plot = plt.plot
semilogy = plt.semilogy
fill = plt.fill
axvspan = plt.axvspan

# Axes limits
xlim = plt.xlim
ylim = plt.ylim
axis = plt.axis

# Titles, axes labels, legend
suptitle = plt.suptitle
title = plt.title
xlabel = plt.xlabel
ylabel = plt.ylabel
legend = plt.legend
grid = plt.grid

# Figure showing
tight_layout = plt.tight_layout
show = plt.show

# Figure saving
savefig = plt.savefig


del plt


# Setup for nicer figures

def init_figure_setup():
    """Setup simplifying code for nicer plots."""
    import matplotlib.pyplot as plt

    # Avoid giving the `figsize` argument when calling `figure`
    plt.rcParams["figure.figsize"] = (10, 6)

    # Avoid calling `tight_layout` after finishing each figure setup
    plt.rcParams["figure.autolayout"] = True

    # Avoid calling `grid` for each figure
    plt.rcParams["axes.grid"] = True

    # Avoid giving the `bbox_inches` argument when calling `savefig`
    plt.rcParams["savefig.bbox"] = "tight"

