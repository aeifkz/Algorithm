
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
