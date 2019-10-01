import matplotlib.pyplot as plt
import numpy as np
import math

N = 2**12 # number of samples
tp = 4 # sampling duration

def signal(t): # signal to be filtered
    pi = math.pi
    sin = math.sin
    return 5*sin(2*pi*50*t) + 3*sin(2*pi*250*t) + 20*sin(2*pi*500*t)

def digital_filter(x): # digital filter implementation
    y = []
    for i in range(len(x)):
        y_1 = y[i-1] if i >= 1 else 0
        y_2 = y[i-2] if i >= 2 else 0
        x_1 = x[i-1] if i >= 1 else 0
        x_2 = x[i-2] if i >= 2 else 0
        # second order Butterworth filter with cut-off frequency of 100Hz
        y.append(1.1429*y_1 - 0.4127*y_2 + 0.067*x[i] + 0.135*x_1 + 0.067*x_2)
    return y

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

def impulse_response(N):
    x = digital_filter([1 if i == 0 else 0 for i in range(N)])
    X = fft(x)
    X_amp = [np.absolute(Xi) for Xi in X]
    M = int(N/2)
    ti = [i*tp/N for i in range(N)]
    fi = [i/tp for i in range(M)]
    X_amp = X_amp[:M]
    plot(1, fi, X_amp, 'f (hertz)', '|X(f)|', 'log', 'Impulse response')

def plot(figure, x, y, xlabel, ylabel, xscale, title):
    plt.figure(figure)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.title(title)
    plt.minorticks_on()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')

impulse_response(N)

x = [round(signal(i*tp/N), 10) for i in range(N)]
ti = [i*tp/N for i in range(N)]
plot(2, ti, x, 't (seconds)', 'x(t)', 'linear', 'Signal in time domain')

_X = fft(x)
X = [round(Xi/N, 10) for Xi in _X]
X_amp = [np.absolute(Xi) for Xi in X]
M = int(N/2)
X_amp = np.array(X_amp[:M])*2
fi = [i/tp for i in range(M)]
plot(3, fi, X_amp, 'f (hertz)', '|X(f)|', 'log', 'Signal in frequency domain')

x_filtered = digital_filter(x)
plot(4, ti, x_filtered, 't (seconds)', 'x(t)', 'linear', 'Filtered signal in time domain')

_X_filtered = fft(x_filtered)
X_filtered = [round(Xi/N, 10) for Xi in _X_filtered]
X_amp_filtered = [np.absolute(Xi) for Xi in X_filtered]
X_amp_filtered = np.array(X_amp_filtered[:M])*2
plot(5, fi, X_amp_filtered, 'f (hertz)', '|X(f)|', 'log', 'Filtered signal in frequency domain')

plt.show()
