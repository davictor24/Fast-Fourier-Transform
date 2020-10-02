import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import math

fs = 2**12  # sample rate
tp = 4  # sampling duration
N = fs*tp  # number of samples


def fft(f):
    Ni = len(f)
    Mi = int(Ni / 2)
    if Mi <= 2:
        return [f[0] + f[1] + f[2] + f[3],
                f[0] - 1j*f[1] - f[2] + 1j*f[3],
                f[0] - f[1] + f[2] - f[3],
                f[0] + 1j*f[1] - f[2] - 1j*f[3]]

    wn = math.cos(2*math.pi/Ni) - 1j*math.sin(2*math.pi/Ni)
    fe = [f[i] for i in range(Ni) if i % 2 == 0]
    fo = [f[i] for i in range(Ni) if i % 2 == 1]
    Fe = fft(fe)
    Fo = fft(fo)
    return [np.around(Fe[i] + (wn**i)*Fo[i], decimals=10) for i in range(Mi)] + [np.around(Fe[i] - (wn**i)*Fo[i], decimals=10) for i in range(Mi)]


recording = sd.rec(N, samplerate=fs, channels=1)
print('Recording...')
sd.wait()  # Wait until recording is finished
print('Done!')
x = [round(float(recording[i]), 10) for i in range(N)]  # input sequence

_X = fft(x)  # discrete Fourier transform
X = [round(Xi/N, 10) for Xi in _X]  # frequency spectrum
X_amp = [np.absolute(Xi) for Xi in X]  # amplitude spectrum

M = int(N/2)
ti = [i*tp/N for i in range(N)]
fi = [i/tp for i in range(M)]
X_amp = np.array(X_amp[:M])*2

plt.subplots_adjust(wspace=0.4, hspace=0.4)

ax1 = plt.subplot(211)
plt.plot(ti, x)
plt.xlabel('t (seconds)')
plt.ylabel('x(t)')
plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='-')
plt.grid(b=True, which='minor', linestyle='--')


ax2 = plt.subplot(212)
plt.plot(fi, X_amp)
plt.xlabel('f (hertz)')
plt.ylabel('|X(f)|')
ax2.set_xscale('log')
plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='-')
plt.grid(b=True, which='minor', linestyle='--')

plt.show()
