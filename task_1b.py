import os
from utils.util import *

init_figure_setup()

# --- 1. SETUP PARAMETERS ---
T = 0.01          # Symbol period
over = 100        # Oversampling
Ts = T / over     # Sampling period

N_inst = 1000     # Number of different implementations of the SP
N_sym = 100 # Number of symbols       
N_bits = 102 # Number of bits      

# --- 2. PULSE DEFINITION (NORMALIZED) ---
# We normalize the pulse to have Unit Energy to match the theoretical derivation
# Energy = amplitude^2 * duration = A^2 * T = 1  => A = 1/sqrt(T)
g_len = over
g = ones(g_len) / sqrt(T)

# --- 3. AXIS SETUP ---
L_impulse = N_sym * over
L_conv = L_impulse + g_len - 1
T_sim = L_conv * Ts
f_axis = fftxaxis(L_conv, Ts)

# --- 4. THEORETICAL PSD ---
# S(f) = sinc^2(fT) * sin^2(2*pi*fT)
# Note: This formula yields values ~0.8 independent of T, requiring the signal 
# to have power scaling with 1/T (provided by the unit energy pulse).
S_theory = (sinc(f_axis * T)**2) * (sin(2 * pi * f_axis * T)**2)

# --- 5. EXPERIMENTAL PSD ---
P_sum = zeros(L_conv)
periodogram_scaler = (Ts**2) / T_sim

for i in range(N_inst):
    # Bits
    b = (rand(N_bits) > 0.5) * 2 - 1
    
    # Symbols (Lag 2)
    s_symbols = 0.5 * (b[2:] - b[:-2]) 
    
    # Upsample
    s_impulse = upsample(s_symbols, over)
    
    # Convolve with normalized pulse
    s_t = conv(s_impulse, g)
    
    # FFT
    S_f = fftshift(fft(s_t))
    
    # Periodogram
    P_inst = periodogram_scaler * (abs(S_f)**2)
    P_sum = P_sum + P_inst

S_exp = P_sum / N_inst

# --- 6. PLOTTING ---
figure()
plot(f_axis, S_exp, label='Experimental PSD')
plot(f_axis, S_theory, 'r--', label='Theoretical PSD')
title('Task 1.b: Power Spectral Density')
xlabel('Frequency (Hz)')
ylabel('PSD')
xlim(-500, 500)
legend()

if not os.path.exists('plots'):
    os.makedirs('plots')
savefig('plots/task_1b.png')