import matplotlib.pyplot as plt
import scipy.io.wavfile as sw
import numpy
import math

N = 128
f = 3

# do inner product to compute diff
def corr(a,b) :
    sum = 0
    for i in range(len(a)) :
        sum += a[i]*b[i]
    return sum


def dft(x) :
    N = len(x)
    X = [0 for m in range(N)]
    for m in range(N) :
        basis = [math.cos(2*math.pi*m/N*n) for n in range(N)]		
        X[m] = corr(x,basis)
    return X


sounds = [ math.cos(2*math.pi*f/N*n) for n in range(N) ]
#fs , sounds = sw.read('/home/aeifkz/viola.wav') 


r = dft(sounds[:])
plt.plot(r)
plt.show()
