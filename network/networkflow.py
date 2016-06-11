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

'''
b = Graph() 
[b.add_v(v) for v in 'stabcdwxyz']

b.add_e('s','a',1)
b.add_e('s','b',1)
b.add_e('s','c',1)
b.add_e('s','d',1)
b.add_e('w','t',1)
b.add_e('x','t',1)
b.add_e('y','t',1)
b.add_e('z','t',1)

b.add_e('a','w',1)
b.add_e('a','x',1)
b.add_e('b','x',1)
b.add_e('b','z',1)
b.add_e('c','y',1)
b.add_e('c','z',1)
b.add_e('d','y',1)

b.reset_flow()
print("Max Flow found from %s to %s is %s" % ( 's','t',b.max_flow('s','t') )  )
'''

'''
g = Graph()

[ g.add_v(v) for v in 'stabcd' ]

g.add_e('s','a',10)
g.add_e('s','c',10)
g.add_e('a','c',2)
g.add_e('a','b',4)
g.add_e('a','d',8)
g.add_e('c','d',9)
g.add_e('d','b',6)
g.add_e('b','t',10)
g.add_e('d','t',10)

#self loop
#g.add_e('c','c',6)

#loop
g.add_e('c','a',6)


g.reset_flow()

print("Max Flow found from %s to %s is %s" % ( 's','t',g.max_flow('s','t') )  )

for k,v in g.vset.items() :
    print(g.vset[k])
'''
