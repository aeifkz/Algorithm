def lzw_enc(source) :
    lzw_dict = { chr(i):chr(i) for i in range(256) }
    cur_size = len(lzw_dict)
    compressed = []
    s = source[0]
    for c in source[1:] :
        print("s={}\tc={}".format(s,c))
        if s+c in lzw_dict : 
            s += c
        else :
            compressed.append(lzw_dict[s])
            print("insert key={}\tvalue={}".format(s+c,cur_size))
            lzw_dict[s+c] = cur_size
            cur_size += 1
            s = c

    compressed.append(lzw_dict[s])

    print("compress len:",len(compressed))

    return compressed


def lzw_dec(compressed) :
    lzw_dict = { chr(i):chr(i) for i in range(256) }
    cur_size = len(lzw_dict)
    uncompressed = []
    uncompressed.append(lzw_dict[compressed[0]])
    s = compressed[0]
    for k in compressed[1:] :
        print("Decoding s={}\tk={}\tcur_size={}".format(s,k,cur_size))
        if k in lzw_dict : 
            decoded_symbol = lzw_dict[k]
        elif k == cur_size :
            decoded_symbol = s + s[0]

        uncompressed.append(decoded_symbol)
        lzw_dict[cur_size] = s + decoded_symbol[0]
        cur_size += 1
        s = decoded_symbol

    print(''.join(uncompressed))

    return uncompressed
   

 

compress = lzw_enc('abcabcabcabcabcabcabc')
print(compress)
print(lzw_dec(compress))
