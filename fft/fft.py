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

def idft(X) :
    N = len(X)
    x = np.array([ X.dot( np.array( [cmath.exp(1j*2*math.pi*n/N*m) 
                for m in range(N)  ]  ) ) for n in  range(N) ])
    return x


sounds = np.array([ math.cos(2*math.pi*f1/N*n) + math.cos(2*math.pi*f2/N*n) for n in range(5,N+5) ])

#fs , sounds = sw.read('/home/aeifkz/viola.wav') 


r = dft(sounds)
s = idft(r)
#plt.plot(r.real)
#plt.plot(r.imag)
plt.plot(sounds)
plt.show()
