import matplotlib.pyplot as plt
import numpy as np
import math

N = 2**5 # number of samples
tp = 4 # sampling duration

def signal(t):
    # pi = math.pi
    # sin = math.sin
    # return 5*sin(2*pi*100*t) + 3*sin(2*pi*50*t) + 20*sin(2*pi*80*t)
    return 1 if t < 1 else 0

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
    
x = [round(signal(i*tp/N), 10) for i in range(N)] # input sequence
_X = fft(x) # discrete Fourier transform
X = [round(Xi/N, 10) for Xi in _X] # frequency spectrum
X_amp = [np.absolute(Xi) for Xi in X] # amplitude spectrum
X_phase = [np.angle(Xi if Xi else 0+0j) for Xi in X] # phase spectrum

M = int(N/2)
ti = [i*tp/N for i in range(N)]
fi = [(i -M + 1)/tp for i in range(N)]
X_amp = X_amp[M+1:] + X_amp[:M+1]
X_phase = X_phase[M+1:] + X_phase[:M+1]

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.subplot(211)
plt.plot(ti, x)
plt.title('x(t)')

plt.subplot(223)
plt.plot(fi, X_amp)
plt.title('|X(f)|')

plt.subplot(224)
plt.plot(fi, X_phase)
plt.title('∠X(f)')

plt.show()
