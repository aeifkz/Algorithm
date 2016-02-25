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


def insert_sort(x) :
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


def bubble_sort(x) :
    for top in range(len(x)) :
        done = True
        for bubble in range( len(x)-1, top , -1) :
            if x[bubble] < x[bubble-1] :
                x[bubble] , x[bubble-1] = x[bubble-1] , x[bubble]
                done = False
            if done :
                return


def merge_sort(x) :

    length = len(x)
    
    if length <=1 : 
        return x
    else :
        y1 = merge_sort(x[0:length//2])
        y2 = merge_sort(x[length//2:])
        y = []

        leny1 = len(y1)
        leny2 = len(y2)
        i1 = 0
        i2 = 0

	#don't use pop(0) , because it is O(n)
	#better use queue
        while i1<leny1 or i2 < leny2 :
            if i1 == leny1 :
                y = y + y2[i2:]
                i2 = leny2
            elif i2 == leny2 :
                y = y + y1[i1:] 
                i1 = leny1
            elif y1[i1] < y2[i2] :
                y.append(y1[i1])
                i1 = i1+1
            else :
                y.append(y2[i2])
                i2 = i2+1
        return y

#in place qsort
def q_sort(x) :
    first = 0
    last = len(x)-1
    qsort_engine(x,first,last)
    return

def qsort_engine(x,first,last) :
    if first < last :
        split_point = partition(x,first,last)
        qsort_engine(x,first,split_point-1)
        qsort_engine(x,split_point+1,last)
    return

def partition(x,first,last) :
    pivot = x[first]
    left = first + 1
    right = last
    done = False
    while not done :
        while left <= right and x[left]<= pivot :
            left += 1

        while right >= left and x[right] >= pivot :
            right -= 1

        if left > right :
            done = True
        else :
            x[left] , x[right] = x[right] , x[left]

    x[first] , x[right] = x[right] , x[first]
    return right


l = [random.random() for n in range(20)]
print(l)
#select_sort(l)
#insert_sort(l)
#bubble_sort(l)
#l = quick_sort(l)
#l = merge_sort(l)
q_sort(l)
print(l)

