def getKey(item) :
    return item[0]

def gen_huff_tree(text) :
    char_list = [ c for c in text ]
    alphabet = set(char_list)
    h_tree = [ ( char_list.count(c), tuple(c) ) for c in alphabet ]

    while len(h_tree) >= 2 :
        h_tree = sorted(h_tree,key=getKey)
        h_tree = [ ( h_tree[0][0]+h_tree[1][0] , (h_tree[0],h_tree[1]) )  ] + h_tree[2:]
    return h_tree


def gen_huff_dict(h_tree_node,code,h_dict) :

    if len(h_tree_node[1])==1 :
        h_dict[h_tree_node[1][0]] = code
    else :
        gen_huff_dict(h_tree_node[1][0],code+'0',h_dict)
        gen_huff_dict(h_tree_node[1][1],code+'1',h_dict)
    return 

def huffman_enc(text_src,h_dict) :
    bit_stream = ''
    for c in text_src :
        bit_stream += h_dict[c]
    return bit_stream

def huffman_dec(bit_stream,h_tree) :
    text_dec = ''
    i=0
    while i < len(bit_stream) :
        cur_node = h_tree[0]
        while len(cur_node[1]) == 2 :
            cur_node = cur_node[1][0] if bit_stream[i] == '0' else cur_node[1][1]
            i += 1
        text_dec += cur_node[1][0]

    return text_dec


text = 'aaabcaa'
print(text)

h_tree = gen_huff_tree(text)
print(h_tree)

h_dict = {}

gen_huff_dict(h_tree[0],'',h_dict)
print(h_dict)

bit_stream = huffman_enc(text,h_dict)
print(bit_stream)

text_dec = huffman_dec(bit_stream,h_tree)
print(text_dec)

