import os
from utils.util import *
from utils.gen_srrc import gen_srrc
from scipy.stats import norm


init_figure_setup()

# --- 1. SYSTEM PARAMETERS ---
# Frequencies and Timing
fc = 800**3          # Carrier: 800 kHz
bw_pass = 80**3      # Passband Bandwidth: 80 kHz
W = bw_pass / 2     # bw_pass = [-W,W]

# Pulse Parameters
half_dur = 4         # Half duration (symbol periods)
rolloff = 0.35        # Roll-off factor

# Symbol Period T derivation:
# Passband BW for SRRC = (1 + rolloff) / T
T = (1 + rolloff) / bw_pass


# Simulation Resolution
over = 200          # Oversampling factor
Ts = T / over       # Sampling period
Fs = 1 / Ts         # Sampling frequency

# Modulation Parameters (4-QAM)
a = 1.0             # Symbol amplitude parameter
# Constellation energy: E[|X|^2] = 2a^2
Es = 2 * (a**2)

# Simulation Size
N_bits = 100000     # 10^5 bits
N_sym = N_bits // 2 # 2 bits per symbol for 4-QAM

# SNR Range (dB)
SNR_dB_range = arange(0, 14, 2) # 0, 2, ..., 12

# --- 2. CHANNEL SCENARIOS (Group 1 from Prompt) ---
# Format: (a0, a1, delay_in_T)
scenarios = [
    (0.5, 0.3, 0.2222),    # Blue: Single Ray (Reference)
    (0.5, 0.3, 0.4444),   # Red: 2nd Ray with Small Delay
    (0.5, 0.3, 0.963)    # Green: 2nd Ray with Large Delay
]
colors = ['b', 'r', 'g']

# --- 3. PULSE GENERATION ---
# Normalized unit energy pulse
phi, t_phi = gen_srrc(T, over, half_dur, rolloff)
# CRITICAL FIX: Normalize pulse energy in the DISCRETE domain
# This ensures Sum(|phi|^2) = 1.
# Now, convolution does not gain/lose energy arbitrarily.
phi = phi / sqrt(sum(abs(phi)**2))


# --- PREPARE PLOTS ---
fig_spec = figure()
ax_mag = subplot(2, 1, 1)
ax_phs = subplot(2, 1, 2)

fig_ber = figure()
ax_ber = subplot(1, 1, 1)

# --- 4. MAIN SIMULATION LOOP ---

for s_idx, (a0, a1, tau_T) in enumerate(scenarios):
    color = colors[s_idx]
    
    # Calculate physical delay t1
    t1 = tau_T * T
    
    # --- A. CHANNEL SPECTRUM H(f) ---
    # Frequency axis for plotting: [-10W, 10W]
    # W = 40 kHz -> Range [-400k, 400k]
    f_plot = arange(-10*W, 10*W, W/50)
    
    # H(f) formula derived from lecture/theory:
    # H(f) = a0 + a1 * exp(-j * 2*pi * fc * t1) * exp(-j * 2*pi * f * t1)
    # The term exp(-j*2*pi*fc*t1) is the baseband phase shift due to passband delay
    H_f = a0 + a1 * exp(-1j * 2 * pi * fc * t1) * exp(-1j * 2 * pi * f_plot * t1)
    
    # Plot Magnitude
    figure(fig_spec.number) # Switch to spectrum figure
    subplot(2, 1, 1)
    # Highlight baseband bandwidth [-W, W] (Yellow)
    if s_idx == 0: # Draw background only once
        axvspan(-W, W, color='yellow', rolloff=0.3, label='BW')
    
    plot(f_plot, abs(H_f), color=color, label=f'({a0}, {a1}, {tau_T}T)')
    
    # Plot Phase
    subplot(2, 1, 2)
    if s_idx == 0:
        axvspan(-W, W, color='yellow', rolloff=0.3)
    
    # Normalize phase by pi for cleaner plot [-1, 1]
    plot(f_plot, angle(H_f)/pi, color=color)

    # --- B. BER SIMULATION ---
    
    # Receiver's Assumption for h (DC gain)
    # h_hat = H(0)
    h_hat = a0 + a1 * exp(-1j * 2 * pi * fc * t1)
    h_abs_sq = abs(h_hat)**2
    
    BER_exp = []
    BER_theo = []
    
    # Generate random bits/symbols (Same for all SNRs to be consistent)
    # 4-QAM: 2 bits -> 1 symbol.
    # Map: 00->(-1-1j), 01->(-1+1j), 10->(1-1j), 11->(1+1j) scaled by 'a'
    # Simplified generation:
    # Real part: +/- a, Imag part: +/- a
    bits_I = 2 * (rand(N_sym) > 0.5) - 1 # -1 or 1
    bits_Q = 2 * (rand(N_sym) > 0.5) - 1 # -1 or 1
    syms = a * (bits_I + 1j * bits_Q)
    
    # Create Baseband Signal
    s_impulse = upsample(syms, over)
    s_baseband = conv(s_impulse, phi)
    
    for SNR_val in SNR_dB_range:
        # Calculate N0
        # SNR_Tx = Es / N0 => N0 = Es / SNR_lin
        SNR_lin = 10**(SNR_val / 10)
        N0 = Es / SNR_lin
        
        # 1. APPLY CHANNEL (Baseband Equivalent)
        # y(t) = a0 * s(t) + a1 * s(t - t1) * exp(-j * 2*pi * fc * t1)
        
        # Delay in samples
        delay_samples = int(round(t1 / Ts))
        phase_shift = exp(-1j * 2 * pi * fc * t1)
        
        # Create delayed copy
        s_delayed = zeros(len(s_baseband), dtype=complex)
        
        if delay_samples == 0:
            # Zero delay: just copy the signal
            s_delayed = s_baseband
        elif delay_samples < len(s_baseband):
            # Normal delay logic
            s_delayed[delay_samples:] = s_baseband[:-delay_samples]
            
        # Combine rays
        r_clean = a0 * s_baseband + a1 * s_delayed * phase_shift
        
        # 2. ADD NOISE
        # Noise variance per sample for simulation = N0 / Ts
        # Complex noise: Real var = sigma^2/2, Imag var = sigma^2/2
        sigma_noise = sqrt(N0)
        noise = (sigma_noise / sqrt(2)) * (randn(len(r_clean)) + 1j * randn(len(r_clean)))
        
        r_received = r_clean + noise
        
        # 3. RECEIVER CHAIN
        # Matched Filter
        # Since phi is real/symmetric, g*(-t) = phi(t)
        y_filtered = conv(r_received, phi)
        
        # Sampling
        # Peak should be at index: (pulse_center) + (delay?)
        # Wait, pulse center is at A*over. Convolution shifts it.
        # We sample at t = kT.
        # The pulse phi generated by gen_srrc is centered at index A*over.
        # Convolving s_impulse (spikes at 0, 100, 200...) with phi
        # puts the peak of the first symbol at index A*over.
        # Convolution doubles the duration, effectively adding delay A*over.
        start_idx = 2 * int(half_dur * over)        
        # Downsample to get symbol estimates
        # We need N_sym samples. Spaced by 'over'.
        y_sampled = y_filtered[start_idx : start_idx + N_sym * over : over]
        
        # 4. DETECTION (Using h_hat)
        # Decision Variable Z = Y * h_hat^*
        Z = y_sampled * conj(h_hat)
        
        # 4-QAM Decision Rule (Task 4.e)
        # Real > 0 -> a, Imag > 0 -> a
        dec_I = 2 * (real(Z) > 0) - 1
        dec_Q = 2 * (imag(Z) > 0) - 1
        
        # Count Errors
        # Total bits = 2 * N_sym
        # Errors = (diff in I) + (diff in Q)
        err_I = sum(dec_I != bits_I)
        err_Q = sum(dec_Q != bits_Q)
        ber = (err_I + err_Q) / (2 * N_sym)
        BER_exp.append(ber)
        
        # 5. THEORETICAL BER
        # For 4-QAM, BER is exactly Q(sqrt(|h|^2 * SNR))
        arg = sqrt(h_abs_sq * SNR_lin)
        ber_th_exact = norm.sf(arg)  # This is BER (p)
        BER_theo.append(ber_th_exact)

    # Plot BER curve for this scenario
    figure(fig_ber.number)
    semilogy(SNR_dB_range, BER_exp, color + '-', label=f'Exp ({a0},{a1},{tau_T})')
    semilogy(SNR_dB_range, BER_theo, color + '--', label=f'Th ({a0},{a1},{tau_T})')

# --- FORMAT PLOTS ---

# Spectrum Figure
figure(fig_spec.number)
subplot(2, 1, 1)
title('$|H(f)|$')
ylabel('$|H(f)|$')
ylim(0, 1)
legend(loc='upper right', fontsize='small')

subplot(2, 1, 2)
title('$arg \; H(f)$ (norm. by $\pi$)')
ylabel('$arg \; H(f)$ / $\pi$')
xlabel('freq (Hz)')
ylim(-1, 1)

if not os.path.exists('plots'):
    os.makedirs('plots')
savefig('plots/task_4g_spectrum1.png')

# BER Figure
figure(fig_ber.number)
title('BER vs SNR$_{T_X}$, 4-QAM, 2-ray channel')
xlabel('SNR$_{T_X}$ (dB)')
ylabel('bit error rate (BER)')
grid(True, which="both", ls="-")
legend()
savefig('plots/task_4g_ber1.png')