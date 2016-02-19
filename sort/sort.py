import random

def select_sort(x) :
    '''select sort O(n^2)'''
    for to_fill in range(len(x)) :
        min_pos = to_fill
        for step in range(to_fill,len(x)) :
            if x[step] < x[min_pos] :
                min_pos = step
        #python swap skill
        x[to_fill] , x[min_pos] = x[min_pos] , x[to_fill]
    return


def bubble_sort(x) :
    '''bubble sort O(n^2)'''
    if len(x) > 1 :
        for bottom in range(1,len(x)) :
            bubble = bottom
            while bubble > 0 and x[bubble] < x[bubble-1] :
                #python swap skill
                x[bubble] , x[bubble-1] = x[bubble-1] , x[bubble]
                bubble -= 1       
        return


def quick_sort(x) :
    '''quick sort O(n*lgn)'''
    if len(x) <= 1 :
        return x
    pivot = x[len(x)//2]
    #easy way to implement quick_sort
    small = [x[n] for n in range(len(x)) if x[n] < pivot]    
    medium = [x[n] for n in range(len(x)) if x[n]==pivot]
    large = [x[n] for n in range(len(x)) if x[n]>pivot]
    return quick_sort(small) + medium + quick_sort(large)




l = [random.random() for n in range(10)]
#select_sort(l)
#bubble_sort(l)
#l = quick_sort(l)
print(l)

