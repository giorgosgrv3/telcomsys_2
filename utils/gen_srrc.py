import numpy as np

def gen_srrc(T, over, A, a):
    """
    phi, t = gen_srrc(T, over, A, a)

    OUTPUT
         phi: truncated SRRC pulse, with parameter T, roll-off
                factor a, and duration 2*A*T
         t:   time axis of the truncated pulse

    INPUT
         T:  Nyquist parameter or symbol period (positive real
               number)
         over: positive integer equal to T/T_s (oversampling
               factor)
         A:  half duration of the pulse in symbol periods
               periods (positive integer)
         a:  roll-off factor (real number between 0 and 1)

       Created using the MATLAB original source code from
       A. P. Liavas, Oct. 2020
    """

    Ts = T / over

    # Create time axis, avoiding division by zero at t = 0
    t = np.arange(-A * T, A * T + Ts/2, Ts) + 1e-8

    if 0 < a <= 1:
        num = np.cos((1 + a) * np.pi * t / T) \
                + np.sin((1 - a) * np.pi * t / T) / (4 * a * t / T)
        denom = 1 - (4 * a * t / T) ** 2
        phi = 4 * a / (np.pi * np.sqrt(T)) * num / denom
    elif a == 0:
        phi = 1 / np.sqrt(T) * np.sin(np.pi * t / T) \
                / (np.pi * t / T)
    else:
        phi = np.zeros(len(t))
        print("Illegal value of roll-off factor")
        return

    return phi, t
