import os
from utils.util import *
init_figure_setup()

N0 = 2 #variance of the white noise
W = 20 #20hz single sided frequency (cause BW is [-w,w])
Th = 0.25 #duration of the impulse response
Fs = 1000 #1000hz sampling frequency
Ts = 1/Fs # = 0.001s

# 1. Input Signal x(t) is the white gaussian noise with zero mean and Rxx(t) = N0 * δ(t)
# in the time interval [0, 1]
t_x = iarange(0, 1, Ts)
sigma_x = sqrt(N0 / Ts) # !! discrete variance = N0 / Ts !! 
x = sigma_x * randn(len(t_x)) # randn() returns a random signal with mean 0 and variance 1

# 2. LTI impulse response h(t)
# If H(f) = rect(f / 2W), then h(t) = 2Wsinc(2Wt)
# duration [-Th/2, Th/2]
t_h = iarange(-Th/2, Th/2, Ts)
h = 2*W*sinc(2*W*t_h)

# 3. output signal y(t)
# approximating continuous convolution with discrete convolution, needs scaling by Ts
y = conv(x, h) * Ts
t_y = convxaxis(t_x, t_h)

# 4. ffts of signals, scaled by Ts for continuous approximation
X_f = fftshift(fft(x)) * Ts 
f_x = fftxaxis(len(x), Ts)

H_f = fftshift(fft(h)) * Ts
f_h = fftxaxis(len(h), Ts)

Y_f = fftshift(fft(y)) * Ts
f_y = fftxaxis(len(y), Ts)


# 5. Plotting
figure()

# Row 1: Time domain
subplot(2, 3, 1)
plot(t_x, x)
title('Input Signal x(t)')
xlabel('Time (s)')
ylabel('Amplitude')

subplot(2, 3, 2)
plot(t_h, h)
title('Impulse Response h(t)')
xlabel('Time (s)')

subplot(2, 3, 3)
plot(t_y, y)
title('Output Signal y(t)')
xlabel('Time (s)')

# Row 2: Frequency domain (Magnitude)
subplot(2, 3, 4)
plot(f_x, abs(X_f))
title('|X(f)|')
xlabel('Frequency (Hz)')

subplot(2, 3, 5)
plot(f_h, abs(H_f))
title('|H(f)|')
xlabel('Frequency (Hz)')
xlim(-3*W, 3*W)

subplot(2, 3, 6)
plot(f_y, abs(Y_f))
title('|Y(f)|')
xlabel('Frequency (Hz)')
xlim(-3*W, 3*W)


if not os.path.exists('plots'):
    os.makedirs('plots')

savefig('plots/task1a.png')
