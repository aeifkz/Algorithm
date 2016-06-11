import numpy as np


class Edge(object) :

    def __init__(self,u,v,c,f) :
        self.source = u
        self.sink = v
        self.capacity = c
        self.flow = f

    def __repr__(self) :
        return "%s --> %s: cap %s flow %s" % (self.source,self.sink,self.capacity,self.flow)


class Graph(object) :
    def __init__(self) :
        self.vset = {}

    def add_v(self,v) :
        self.vset[v] = []

    def get_e(self,v) :
        return self.vset[v]

    def add_e(self,u,v,c=0) :
        if(u==v) :
            pass
            #raise ValueError("u == v")

        edge = Edge(u,v,c,0)
        redge = Edge(v,u,0,0)
        edge.redge = redge
        redge.redge = edge
        self.vset[u].append(edge)
        self.vset[v].append(redge)

    def reset_flow(self) :
        for v in self.vset :
            for edge in self.vset[v] :
                edge.flow = 0 


    def find_path(self,source,sink,path) :
        if (source == sink) :
            return path 
        else :
            for edge in self.get_e(source) :
                res = edge.capacity - edge.flow
                if res > 0 and edge not in path and edge.redge not in path :
                    result = self.find_path(edge.sink,sink,path+[edge])
                    if result != None :
                        return result


    def max_flow(self,source,sink) :
        path = self.find_path(source,sink,[])
        while path != None :
            flow = min( [ edge.capacity - edge.flow for edge in path ])
            print(flow,path)
            for edge in path :
                edge.flow += flow
                edge.redge.flow -= flow
            path = self.find_path(source,sink,[])
        return sum( edge.flow for edge in self.get_e(source) )


    def b_matching(self,source,sink) :
        path = self.find_path(source,sink,[])
        while path != None :
            flow = min( [ edge.capacity - edge.flow for edge in path ])
            print(flow,path)
            for edge in path :
                edge.flow += flow
                edge.redge.flow -= flow
            path = self.find_path(source,sink,[])
        match = [  (e.sink , m.sink) 
                   for e in self.get_e(source) if e.flow == 1
                   for m in self.get_e(e.sink) if m.flow == 1   ]
        return match

    def v_cover(self, source , sink , match) :
        L  = {e.sink for e in self.get_e(source)}
        R  = {e.sink for e in self.get_e(sink)}
        ML = {t[0] for t in match}
        MR = {t[1] for t in match}
        U  = L - ML
        Z  = set()
        Q  = list(U)
        while Q :
            q = Q.pop()
            Z.add(q)
            if q in L :
                for e in self.vset[q] :
                    if e.sink in R and e.flow != 1 :
                        Q.append(e.sink)
                    elif q in R :
                        for e in self.vset[q] :
                            if e.sink in ML and e.redge.flow == 1 :
                                Q.append(e.sink)
        K = (L-Z).union(R.intersection(Z))
        return K
                        

def min_cover(a) :

    (Nh,Nw) = a.shape

    h = Graph()
    h.add_v('s')
    h.add_v('t')
    for i in range(Nh) :
        h.add_v('L'+str(i))
        h.add_e('s','L'+str(i),1)
    for j in range(Nw) :
        h.add_v('R'+str(j))
        h.add_e('R'+str(j),'t',1)

    for i in range(Nh) :
        for j in range(Nw) :
            if a[i,j] == 0 :
                h.add_e('L'+str(i),'R'+str(j),1)

    h.reset_flow()
    match = h.b_matching('s','t')
    k = h.v_cover('s','t',match) 

    print(k)

    return k


def matching(a) :
    (Nh , Nw) = a.shape
    for i in range(Nh) :
        a[i , :] -= np.min(a[i,:])

    for j in range(Nw) :
        a[ : , j] -= np.min(a[:,j])

    while True :
        cover = min_cover(a)
        if len(cover) == Nh :
            print("a:",a)
            return a
        else :
            h_lines = []
            v_lines = []
            min_uncover = min( a[i,j] 
                               for i in range(Nh) if i not in h_lines 
                               for j in range(Nw) if j not in v_lines 
                                                                        )
            for i in range(Nh) :
                if i not in h_lines : 
                    a[i , :] -= min_uncover
            
            for j in range(Nw) :
                if j in v_lines : 
                    a[ : , j] -= min_uncover


a = np.array(
             [[90,75,75,80],
              [35,85,55,65],
              [125,95,90,105] ,
              [45,110,95,115]]   )

matching(a)
