import numpy as np




def simplex(t_in,sc_in) :

    np.set_printoptions(precision=2 , suppress = True)

    t = np.copy(t_in)
    s = np.copy(sc_in)
    (num_row,num_col) = t.shape
    num_ineq = num_row - 1
    num_slack = num_ineq
    num_var = num_col - num_slack - 1 - 1
    print("Init Tableau\n",t)
    print("Init Swap\n",sc)

    
    while t[-1,np.argmin(t[-1,:]) ] < 0  :
        pivotcol = np.argmin(t[-1,:]) 
        pivotrow = np.argmin(t[:num_ineq,-1]/t[:num_ineq,pivotcol])

        t[pivotrow,:] = t[pivotrow,:] / t[pivotrow , pivotcol]
        for r in range(num_row) :
            if r != pivotrow :
                t[r,:] = t[r,:] + t[pivotrow,:] * (-1*t[r,pivotcol])
        sc[pivotrow] = pivotcol
        print("\nTableau\n",t)
        print("Swap Col\n",sc)

    var_list = np.zeros(num_var)

    for i in range(len(sc)-1) :
        if sc[i] in range(num_var) :
            var_list[sc[i]] = t[i,-1]

    return var_list , t , sc


t = np.array( [ 
                [ 2.0, 3.0,  2.0 , 1.0 , 0.0 , 0.0 , 1000.0] , 
                [ 1.0, 1.0,  2.0 , 0.0 , 1.0 , 0.0 ,  800.0] , 
                [-7.0,-8.0,-10.0 , 0.0 , 0.0 , 1.0 ,    0.0] , 
                                                                ])

sc = np.array( [ 3 , 4 , 5 ] )


var_setting , t_ret , sc_ret = simplex(t,sc)


print("Var Seeting\n",var_setting)
print("Tableau Returned\n",t_ret)
print("SC Returned\n",sc_ret)
