
class Edge(object) :
    def __init__(self,u,v) :
        self.source = u 
        self.sink = v

    def __repr__(self) :
        return "%s --> %s" % (self.source,self.sink)


class Graph(object) :
    def __init__(self) :
        self.vset = {}
        self.inadj = {}
        self.outadj = {}
        self.indeg = {}
        self.outdeg = {}
        self.pr = {}

    def add_v(self,name) :
        self.vset[name] = name
        self.inadj[name] = []
        self.outadj[name] = []
        self.indeg[name] = 0
        self.outdeg[name] = 0
        self.pr[name] = 0.0

    def add_e(self,u,v) :
        edge = Edge(u,v)
        self.outadj[u].append(edge)
        self.inadj[v].append(edge)
        self.outdeg[u] += 1
        self.indeg[v] += 1

    def init_pr(self) :
        for v in self.vset :
            self.pr[v] = 1.0/len(self.vset)


    # sigma , like threshold to stop
    def page_rank(self,sigma=0.01,df=0.85) :
        max_delta = 1.0
        
        while max_delta > sigma :
            print("Current PR",self.pr,"\n")            
            max_delta = 0.0
            for u in self.vset :
                neigh_contr = 0.0
                for v in [ edge.source  for edge in self.inadj[u] ] :
                    neigh_contr += self.pr[v] / self.outdeg[v]
                newpr = df*neigh_contr + (1.0-df)/len(self.vset)
                self.pr[u] , delta = newpr , abs(newpr-self.pr[u])
                if delta > max_delta :
                    max_delta = delta
        return



g = Graph()

[g.add_v(v) for v in 'sopqrt']


for edge in ['so','sp','op','pr','oq','qr','qt','rt','rs','tr'] :
    g.add_e(edge[0],edge[1]) 

g.init_pr()
g.page_rank()


print("add edge.\n")

for edge in ['st','ot'] :
    g.add_e(edge[0],edge[1]) 

g.page_rank()



