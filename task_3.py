import os
# Import everything from your provided utils
from utils.util import *
# Also import the pulse generator
from utils.gen_srrc import gen_srrc 

# Initialize plot settings
init_figure_setup()

# --- 1. PARAMETERS ---
fc = 6.0         # Carrier frequency (Hz)
T = 1.0          # Symbol period (s)
over = 100       # Oversampling factor
Ts = T / over    # Sampling period
Fs = 1 / Ts      # Sampling frequency

M = 5            # 5-FSK
symbols_indices = arange(M) # 0, 1, 2, 3, 4

# Symmetric Mapping: sm = m - 2 -> {-2, -1, 0, 1, 2}
s_mapped = symbols_indices - 2

# Two scenarios for Delta F
# Case 1: 1/T (Orthogonal)
# Case 2: 1/2T (Tighter, Interference)
DeltaF_values = [1/T, 1/(2*T)]
DeltaF_labels = ['1/T', '1/2T']

# --- 2. PULSE DEFINITION ---
# Rectangular pulse phi(t) constant in [0, T)
# Normalized to unit energy: 1/sqrt(T)
phi_amp = 1/sqrt(T)
phi = ones(over) * phi_amp 
# Pulse time axis (relative to 0)
t_phi = arange(0, T, Ts)

# --- 3. MAIN SIMULATION LOOP ---
# Loop 1: Scenarios (Delta F)
for d_idx, DeltaF in enumerate(DeltaF_values):
    label_DF = DeltaF_labels[d_idx]
    
    # Loop 2: Each Symbol (m = 0 to 4)
    for m in range(M):
        # Current symbol's frequency multiplier
        sm = s_mapped[m]
        
        # --- A. TRANSMITTER ---
        
        # 1. Baseband Pulse Generation for Symbol m
        # g_m(t) = phi(t) * exp(j * 2*pi * sm * DeltaF * t)
        g_m_t = phi * exp(1j * 2 * pi * sm * DeltaF * t_phi)
        
        # Baseband signal s(t) (Single symbol K=1)
        s_baseband = g_m_t
        t_s = t_phi # Time axis for the signal
        
        # 2. Upconversion to Passband sBP(t)
        # sBP(t) = Re{ s(t) * 2 * exp(j * 2*pi * fc * t) }
        s_bp = real(s_baseband * 2 * exp(1j * 2 * pi * fc * t_s))
        
        
        # --- B. CHANNEL (Ideal) ---
        r_bp = s_bp
        
        
        # --- C. RECEIVER ---
        
        # 1. Unfiltered Downconversion
        # r(t) = rBP(t) * exp(-j * 2*pi * fc * t)
        r_unfiltered = r_bp * exp(-1j * 2 * pi * fc * t_s)
        
        # 2. Filter Bank (Matched Filters)
        filter_outputs = []
        filter_time_axes = []
        
        for m_prime in range(M):
            sm_prime = s_mapped[m_prime]
            
            # Generate the pulse for frequency m'
            g_m_prime_t = phi * exp(1j * 2 * pi * sm_prime * DeltaF * t_phi)
            
            # Matched Filter Impulse Response: h(t) = g*(-t)
            # using 'conj' and 'flip' from utils
            h_filter = conj(flip(g_m_prime_t))
            
            # Convolve: y = r * h
            # Scale by Ts for continuous approximation
            y_out = conv(r_unfiltered, h_filter) * Ts
            filter_outputs.append(y_out)
            
            # Create time axis for the filter output
            # Convolution of length L with length L results in 2L-1 points
            # Axis needs to reflect the shift. 
            # We can use convxaxis logic or manual construction.
            # Manual construction consistent with previous logic:
            t_y = arange(0, len(y_out)) * Ts - (T - Ts)
            filter_time_axes.append(t_y)

        
        # --- D. PLOTTING ---
        figure()
        
        # Layout Title
        suptitle(f'$\Delta F = {label_DF}$, symbol: $m={m}$ ($s_m={sm}$)')
        
        # 1. Baseband s(t)
        subplot(4, 1, 1)
        plot(t_s, real(s_baseband), 'b', linewidth=1, label='Real')
        plot(t_s, imag(s_baseband), 'r', linewidth=1, label='Imag')
        title('baseband s(t)')
        ylabel('amplitude')
        
        # 2. Upconverted sBP(t)
        subplot(4, 1, 2)
        plot(t_s, s_bp, 'b', linewidth=1)
        title('upconverted $s_{BP}(t)$')
        ylabel('amplitude')
        
        # 3. Downconverted r(t)
        subplot(4, 1, 3)
        plot(t_s, real(r_unfiltered), 'b', linewidth=1)
        plot(t_s, imag(r_unfiltered), 'r', linewidth=1)
        title('downconverted (unfiltered) r(t)')
        ylabel('amplitude')
        
        # 4. Filter Bank Outputs (Row 2)
        # Calculate max value for scaling axes uniformly
        # Using 'max' and 'abs' from utils
        max_val = 0
        for out in filter_outputs:
            current_max = max(abs(out))
            if current_max > max_val:
                max_val = current_max
        
        # Plotting the 5 filters
        for i in range(5):
            subplot(4, 5, 16 + i)
            plot(filter_time_axes[i], abs(filter_outputs[i]), 'b', linewidth=1.2)
            title(f'filter {i}\n($s_{{m\'}}={s_mapped[i]}$)')
            xlabel('time (s)')
            
            # Set uniform y-limits to make comparison easy
            if max_val > 0:
                ylim(0, max_val * 1.1)

        # Save the figure
        filename = f'plots/task_3g_DF{d_idx+1}_Sym{m}.png'
        if not os.path.exists('plots'):
            os.makedirs('plots')
        savefig(filename)