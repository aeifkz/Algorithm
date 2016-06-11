from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from fft import *
from compress import *



def dct(img_mat) :
    N=8
    C = [1.0/math.sqrt(2.0)] + [1.0]*(N-1)
    coeff_mat = np.zeros((N,N),dtype=int)
    for r in range(N) :
        for c in range(N) :
            coeff_mat[r,c] = int( 1/4*C[r]*C[c]* 
                                  sum( [ img_mat[x,y]* 
                                         math.cos(math.pi*(2*x+1)*r/16)* 
                                         math.cos(math.pi*(2*y+1)*c/16)
                                         for x in range(N)  for y in range(N) ]  )  )
    return coeff_mat




def idct(icoeff_mat) :
    pass


def zigzag(q_coeff_mat) :
    N = 8
    q_coeff_list = []
    for i in range(N) :
        if i%2 == 0 :
            x,y = i,0
            for k in range(i+1) :
                q_coeff_list.append(q_coeff_mat[x,y])
                x,y = x-1 , y+1
        else :
            x,y = 0,i
            for k in range(i+1) :
                q_coeff_list.append(q_coeff_mat[x,y])
                x,y = x+1 , y-1

    for j in  range(1,N) :
        if j%2 != 0 :
            x,y = N-1 , j
            for k in range(N-j) :
                q_coeff_list.append(q_coeff_mat[x,y])
                x,y = x-1 , y+1
        else :
            x,y = j,N-1
            for k in range(N-j) :
                q_coeff_list.append(q_coeff_mat[x,y])
                x,y = x+1 , y-1

    return q_coeff_list




def q(coeff_mat,q_mat) :
    N=8    
    q_coeff_mat = np.zeros(coeff_mat.shape,dtype=int)
    q_coeff_mat = (coeff_mat/q_mat + 0.5) //1
    '''
    q_coeff_mat = coeff_mat // q_mat ;
    #if -0.001 => -1 , reverse will get large number , so add 1
    
    for i in range(N):
			for j in range(N) :
            if q_coeff_mat[i,j] < 0 :
                q_coeff_mat[i,j] += 1
    '''
    return q_coeff_mat


def iq(iq_coeff_mat,q_mat) :
    return iq_coeff_mat*q_mat ;




image_src0 = misc.imread('lenna.png')

print(image_src0.shape , image_src0.dtype)

H,W,O = image_src0.shape
image_src = image_src0[:H//16,:W//16,0]

H = H//16
W = W//16

print(image_src)

N=8

q_mat = np.array([
                   [16,11,10,16,24,40,51,61] , 
                   [12,12,14,19,26,58,60,55] ,
                   [14,13,16,24,40,57,69,56] ,
                   [14,17,22,29,51,87,80,62] ,
                   [18,22,37,56,68,109,103,77] ,
                   [24,35,55,64,81,104,113,92] ,
                   [49,64,78,87,103,121,120,101] ,
                   [72,92,95,98,112,100,103,99]
                 ])

#fft2d(image_src)


decode_dict = {} 
dc_list = []


for r in range(0,H//N) :
    for c in range(0,W//N) :
        image_mat = image_src[r*N:(r+1)*N,c*N:(c+1)*N]
        coeff_mat = dct(image_mat)
        q_coeff_mat = q(coeff_mat,q_mat)
        zz = zigzag(q_coeff_mat)
        dc_list.append(zz[0])
        ac_run = rl_enc(zz[1:])
        ac_run_collection.extend(ac_run)
        decode_dict[(r,c)] = {
                                'img_mat': image_mat,
                                'coeff_mat': coeff_mat,
                                'q_coeff_mat': q_coeff_mat,
                                'zz': zz ,
                                'ac_run': ac_run
                                                             }


        #DC
        dc_diff_list = [dc_list[0]] + [dc_list[i]-dc_list[i-1] for i in range(1,len(dc_list)) ]
        
        dc_tree = gen_huff_tree(dc_diff_list) 

        dc_dict = {}
        gen_huffman_dict(dc_tree[0],dc_dict)
        
        dc_code = huffman_enc(dc_dict,dc_diff_list)
        dc_diff_decoded = huffman_dec(dc_dict,dc_code)

        dc_list_decoded = [dc_diff_decoded[0]]
        for i in range(i,len(dc_list_decoded)) :
            dc_list_decoded.append(dc_list_decoded[-1]+dc_diff_decoded[i])



        #Build Huff Tree for AC runs
        rl_tree = gen_huff_tree(ac_run_collection) 

        rl_dict = {}
        gen_huffman_dict(rl_tree[0],rl_dict)


for r in range(0,H//N) :
    for c in range(0,W//N) :
        
        ac_run_encoded = huffman_enc( rl_dict , decode_dict[(r,c)]['ac_run']  )
        ac_run_decoded = huffman_dec( rl_tree , ac_run_encoded )

        izz = rl_dec( ac_run_decoded )
        izz.insert( 0 , dc_list[ r*(W//N) + c ] )

        iq_coeff_mat = decode_dict[(r,c)]['q_coeff_mat']

        icoeff_mat = iq(iq_coeff_mat,q_mat)
        reconstructured = idct[icoeff_mat]

        decode_dict[(r,c)]['ac_run_encoded'] = ac_run_encoded
        decode_dict[(r,c)]['ac_run_decoded'] = ac_run_decoded
        decode_dict[(r,c)]['izz'] = izz
        decode_dict[(r,c)]['iq_coeff_mat'] = iq_coeff_mat
        decode_dict[(r,c)]['icoeff_mat'] = icoeff_mat
        decode_dict[(r,c)]['reconstructed'] = reconstructured


