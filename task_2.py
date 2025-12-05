import os
from utils.util import *
from utils.gen_srrc import gen_srrc 

init_figure_setup()

fc = 2.0
T = 1.0 
oversampling = 10 
half_duration = 5 #half duration of pulse in symbol periods
rolloff = 0.4 #roll-off

Ts = T / oversampling 

##### transmitter #####

#define s0,s1,s2,s3,s4
symbols = array([-1-1j, 1-1j, 1+1j, -1-1j, 1+1j])
N_sym = len(symbols) # = 5 symbols 

#g(t)
phi, t_phi = gen_srrc(T, oversampling, half_duration, rolloff)

#s(t)
s_impulse = upsample(symbols, oversampling) #impulse train sum_i{s(t)*δ(t-iT)}
s_baseband = conv(s_impulse, phi)*Ts #conv with g(t) to make sum_i{s(t)*g(t-iT)}

#time axis for s(t)
t_start = t_phi[0]
t_s = t_start + arange(len(s_baseband))*Ts

#sBP(t)
s_bp = real(s_baseband * 2 * exp(1j * 2 * pi * fc * t_s))

#channel ideal
r_bp = s_bp

###### receiver ######

#demodulation - unfiltered downconversion
# r(t) = sBP(t) * e^(-j2πfct)
r_unfiltered = s_bp * exp(-1j * 2 * pi * fc * t_s)

#match filter g(-t), also g(t) is the same since the pulse is srrc
g_matched = flip(phi) 

# convolve with match filter, scale by Ts 
y_filtered = conv(r_unfiltered, g_matched) * Ts

# time axis for filtered r(t)
# Convolution adds the starting times: t_start_y = t_start_signal + t_start_filter
t_y_start = t_s[0] + t_phi[0]
t_y = t_y_start + arange(len(y_filtered)) * Ts


### fourier ###

# baseband
S_f = fftshift(fft(s_baseband)) * Ts
f_s = fftxaxis(len(s_baseband), Ts)

# passband
X_f = fftshift(fft(s_bp)) * Ts
f_x = fftxaxis(len(s_bp), Ts)

# downconverted unfiltered
R_f = fftshift(fft(r_unfiltered)) * Ts
f_r = fftxaxis(len(r_unfiltered), Ts)

# filtered
Y_f = fftshift(fft(y_filtered)) * Ts
f_y = fftxaxis(len(y_filtered), Ts)

figure()

# first row -> time domain
# baseband
subplot(2, 4, 1)
plot(t_s, real(s_baseband), 'b', linewidth=0.7)
plot(t_s, imag(s_baseband), 'r', linewidth=0.7)
title('baseband s(t)')
xlabel('Time (s)')

# passbnad
subplot(2, 4, 2)
plot(t_s, s_bp, 'b', linewidth=0.7)
title('bandpass sBP(t)')
xlabel('Time (s)')

# 3. Downconverted
subplot(2, 4, 3)
plot(t_s, real(r_unfiltered), 'b', linewidth=0.7)
plot(t_s, imag(r_unfiltered), 'r', linewidth=0.7)
title('downconverted r(t)')
xlabel('Time (s)')

# filtered downconverted
subplot(2, 4, 4)
plot(t_y, real(y_filtered), 'b', linewidth=0.7)
plot(t_y, imag(y_filtered), 'r', linewidth=0.7)
title('filtered r(t)')
xlabel('Time (s)')


# second row -> magnitudes in the frequency domain
#|S(f)|
subplot(2, 4, 5)
plot(f_s, abs(S_f), 'b', linewidth=0.7)
title('|S(f)|')
xlabel('Freq (Hz)')
xlim(-fc*3, fc*3) 

#|sBP(f)|
subplot(2, 4, 6)
plot(f_x, abs(X_f), 'b', linewidth=0.7)
title('|sBP(f)|')
xlabel('Freq (Hz)')
xlim(-fc*3, fc*3)

#|R(f)|
subplot(2, 4, 7)
plot(f_r, abs(R_f), 'b', linewidth=0.7)
title('downcoverted |R(f)|')
xlabel('Freq (Hz)')
xlim(-fc*3, fc*3)

# |Y(f)|
subplot(2, 4, 8)
plot(f_y, abs(Y_f), 'b', linewidth=0.7)
title('filtered |R(f)|')
xlabel('Freq (Hz)')
xlim(-fc*3, fc*3)

if not os.path.exists('plots'):
    os.makedirs('plots')
savefig('plots/task_2a.png')