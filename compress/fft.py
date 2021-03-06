import matplotlib.pyplot as plt
import scipy.io.wavfile as sw
import numpy as np
import cmath
import math

N = 128
f1 = 3
f2 = 9

# do inner product to compute diff
def corr(a,b) :
    sum = 0
    for i in range(len(a)) :
        sum += a[i]*b[i]
    return sum


def dft(x) :
    N = len(x)
    #X = [0 for m in range(N)]
    X = np.zeros(N,dtype=complex)
    for m in range(N) :
        #basis = np.array([math.cos(2*math.pi*m/N*n) for n in range(N)])
        #better basis
        basis = np.array([ cmath.exp(-1j*2*math.pi*m/N*n) for n in range(N) ])
        X[m] = corr(x,basis)
    return X

def fft(x) :
    N = len(x)
    X = np.zeros(N,dtype=complex)
    half_N = N//2

    if N==1 :
        X[0] = x[0]
    else :
        x_even = np.array( [ x[2*k] for k in range(half_N) ] , dtype=complex )
        x_odd = np.array( [ x[2*l+1] for l in range(half_N) ] , dtype=complex )
        X_even = fft(x_even)
        X_odd = fft(x_odd)
        X_odd_mult = np.array( [ X_odd[m] * cmath.exp(-1j*2*math.pi/N*m) for m in range(half_N) ])

        for m in range(half_N) :
            X[m] = X_even[m % half_N] +  X_odd_mult[m%half_N]

        for m in range(half_N,N) :
            X[m] = X_even[m % half_N] - X_odd_mult[m%half_N]

    return X


def idft(X) :
    N = len(X)
    x = np.array([ X.dot( np.array( [cmath.exp(1j*2*math.pi*n/N*m) 
                for m in range(N)  ]  ) ) for n in  range(N) ])
    return x

def ifft(X) :
    return ifft_engine(X) / len(X)


def ifft_engine(X) :
    N = len(X)
    x = np.zeros(N,dtype=complex)
    half_N = N//2

    if N==1 :
        x[0] = X[0]
    else :
        X_even = np.array( [ X[2*k] for k in range(half_N) ] , dtype=complex )
        X_odd = np.array( [ X[2*l+1] for l in range(half_N) ] , dtype=complex )
        x_even = ifft_engine(X_even)
        x_odd = ifft_engine(X_odd)
        x_odd_mult = np.array( [ x_odd[n] * cmath.exp(1j*2*math.pi/N*n) for n in range(half_N) ])

        for n in range(half_N) :
            x[n] = x_even[n % half_N] +  x_odd_mult[n%half_N]

        for n in range(half_N,N) :
            x[n] = x_even[n % half_N] - x_odd_mult[n%half_N]

    return x



def old_ifft(X) :
    N = len(X)
    x = np.zeros(N,dtype=complex)
    half_N = N//2

    if N==1 :
        x[0] = X[0]
    else :
        X_even = np.array( [ X[2*k] for k in range(half_N) ] , dtype=complex )
        X_odd = np.array( [ X[2*l+1] for l in range(half_N) ] , dtype=complex )
        x_even = ifft(X_even)
        x_odd = ifft(X_odd)

        for n in range(N) :
            x[n] = x_even[n % half_N] + cmath.exp(1j*2*math.pi/N*n)*x_odd[n % half_N]

    return x


def fft2d(f) :
    (Nr,Nc,o) = f.shape
    F = np.zeros((Nr,Nc),dtype=complex)
    for m in range(Nr) :
        F[m,:] = fft(f[m,:])
        
    for n in range(Nc) :
        F[:,n] = fft(F[:,n])
    return (F)
   


