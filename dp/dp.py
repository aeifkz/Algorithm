import numpy as np

def fib(n) :
	if n==0 :
		return 0
	elif n==1 :
		return 1
	else :
		return fib(n-1) + fib(n-2)


def fib_dp(n) :

	sol = [-1 for i in range(n+1)]
	sol[0] = 0
	sol[1] = 1
	
	for i in range(2,n+1) :
		sol[i] = sol[i-1] + sol[i-2]

	print(sol)

	return sol[-1]


def lcs_bf(x,y) :
	if len(x) == 0 or len(y) == 0 :
		return 0
	else :
		if x[-1] == y[-1] :
			return lcs_bf(x[:-1],y[:-1]) + 1
		else :
			return max(lcs_bf(x[:-1],y),lcs_bf(x,y[:-1]))

def lcs_dict(x,y) :
	dict = {} 
	return lcs_dict_engine(x,y,dict)

def lcs_dict_engine(x,y,dict) :

	key = str(len(x))+'+'+str(len(y)) 

	if len(x) == 0 or len(y) == 0 :
		return 0 

	if key in dict :
		return dict[key]
	else :
		if x[-1] == y[-1] :
			sol = lcs_dict_engine(x[:-1],y[:-1],dict) + 1
		else :
			sol = max ( lcs_dict_engine(x[:-1],y,dict) , lcs_dict_engine(x,y[:-1],dict) )

		dict[key] = sol 
		
		return sol

def lcs_array(x,y) :
    dp_array = np.zeros( (len(x)+1,len(y)+1) )
    for r in range(1,len(x)+1) :
        for c in range(1,len(y)+1) :
            if (x[r-1]==y[c-1]) :
                dp_array[r,c] = dp_array[r-1,c-1]+1
            else :
                dp_array[r,c] = max(dp_array[r,c-1],dp_array[r-1,c])

    print(dp_array)
    return dp_array[r,c]


def text_just(text_list,start,end,width) :
    if start == end :
        return 0
    min_cost = 9999999999
    min_cut_point = start
    candidate = start
    usage = -1
    while(candidate<end and usage + len(text_list[candidate])+1<=width) :
        usage = usage + 1 + len(text_list[candidate])
        line_cost = (width-usage)**2
        print(candidate , line_cost)
        curr_cost = line_cost + text_just(text_list,candidate+1,end,width)
        if curr_cost < min_cost :
            min_cost = curr_cost
            min_cut_point = candidate
        candidate += 1
    print("Cut at word ",text_list[candidate-1],candidate-1)
    return min_cost


def chain_mat(x) :
    if(len(x)<=1) :
        return 0
    return min( [ chain_mat(x[:i+1]) + chain_mat(x[i+1:]) + x[0][0]*x[i][1]*x[-1][1] 
                                                            for i in range(0,len(x)-1)  ] )

def chain_mat_dict(x) :
    sol_dict = {}
    return chain_mat_dict_engine(x,0,len(x)-1,sol_dict)


def chain_mat_dict_engine(x,start,end,sol_dict) :
    if start == end :
        return 0
    
    key = str(start)+'+'+str(end)

    if key in sol_dict :
        return sol_dict[key]
    else :
        sol_dict[key] =  min( [ chain_mat_dict_engine(x,start,i,sol_dict) +
                                chain_mat_dict_engine(x,i+1,end,sol_dict) +
                                x[start][0]*x[i][1]*x[end][1] 
                                for i in range(start,end) ] )
        #print("key and value:",key,sol_dict[key])
        return sol_dict[key]


def chain_mat_loop(x) :
    N = len(x)

    if N<=1 :
        return 0

    dp_array = np.zeros((N,N))

    for r in range(N-1-1,-1,-1) :
        for c in range(r+1,N) :
            dp_array[r][c] = min( [ dp_array[r][i] +
                                    dp_array[i+1][c] +
                                    x[r][0]*x[i][1]*x[c][1] 
                                            for i in range(r,c) ] )

    print(dp_array)

    return dp_array[0][N-1]


INFINITY = 10**10


class Edge(object) :
    def __init__(self,u,v,c) :
        self.source = u
        self.sink = v
        self.cost = c

    def __repr__(self) :
        return "%s-->%s: cost %s" % (self.source , self.sink , self.cost)


class Graph(object) :

    def __init__(self) :
        self.inset = {}
        self.outset = {}
        self.min_cost = {} 
        self.min_edge = {}
        return

    def __repr__(self) :
        return "Graph has %s vertices" % (len(self.inset))
    

    def add_vertex(self,v) :
        self.inset[v] = []
        self.outset[v] = []

    def add_edge(self,u,v,c=0) :
        edge = Edge(u,v,c)
        self.inset[v].append(edge)
        self.outset[u].append(edge)

    def reset_path(self,source) :
        for u in self.inset.keys() :
            self.min_cost[u] = INFINITY
        self.min_cost[source] = 0

    def s_path_bf(self,source,sink,cost,path) :
        if source == sink :
            return cost , path
        else :
            min_cost , min_path = INFINITY , []
            for edge in self.inset[sink] :
                new_cost , new_path = self.s_path_bf(source,edge.source,cost+edge.cost,path+[edge])
                if new_cost < min_cost :
                    min_cost , min_path = new_cost , new_path

            return min_cost , min_path


    def s_path_dp(self,source,sink) :

        if source == sink :
            return 0
        else :
            min_cost , min_edge = INFINITY , None
            for edge in self.inset[sink] :
                if self.min_cost[edge.source] < INFINITY :
                    new_cost = self.min_cost[edge.source] + edge.cost
                else :
                    new_cost = self.s_path_dp(source,edge.source)+edge.cost

                if new_cost < min_cost :
                    min_cost , min_edge = new_cost , edge

            self.min_cost[sink] = min_cost
            self.min_edge[sink] = min_edge

            return min_cost
        

g = Graph()
[g.add_vertex(v) for v in 'sopqrt']

g.add_edge('s','o',3)
g.add_edge('s','p',3)
g.add_edge('o','p',2)
g.add_edge('p','r',2)
g.add_edge('o','q',3)
g.add_edge('q','r',4)
g.add_edge('q','t',2)
g.add_edge('r','t',1)

'''
g.add_edge('s','o',1)
g.add_edge('s','p',5)
g.add_edge('o','p',1)
g.add_edge('p','r',1)
g.add_edge('o','q',5)
g.add_edge('r','q',1)
g.add_edge('q','t',1)
g.add_edge('r','t',5)
'''

g.reset_path('s')
print(g.min_cost)
#print(g.s_path_bf('s','t',0,[]))
print(g.s_path_dp('s','t'))
print(g)

#print(g.inset)
#print(g.outset)
#print(g.min_cost)

#x = [(3,20),(20,100),(100,30),(30,50),(50,1000),(1000,100),(100,3)]

#x = [(2,5),(5,100),(100,3),(3,9)]
#print(chain_mat_dict(x))
#print(chain_mat_loop(x))
   

#text = 'It was the best of time'.split()
#width = 10
#print(text_just(text,0,len(text),width))

