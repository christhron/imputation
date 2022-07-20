import itertools
import numpy as np
####################

# a function to assign df columns to an index
def create_dic (keys, values):
    dicts = {}
    for i in range(len(values)):
        dicts[keys[i]] = values[i]
        
    return dicts    

#######################################

# a function to create a pairs of two of a given list
def create_comb (cols):
    binary_comb = list(itertools.combinations(cols, 2))
    for i in cols:
        binary_comb.append((i,i))
    return binary_comb

##########################


def create_symMatrix (df, binary_comb, w, h):
    w, h = 8, 8
    dicts = create_dic (keys = df.columns[2:], values = range( len(df.columns[2:]) ) )
    
    Matrix = [[0 for x in range(w)] for y in range(h)] 
    for i in binary_comb:    
        df_tmp = df.copy()
        df_tmp = df_tmp[[i[0], i[1]]] 
        df_tmp = df_tmp.dropna() 

        num = np.array(df_tmp.astype(float).cov())[0][1] # will this be correct for the same var ?
        Matrix[ dicts[i[0]]][dicts[i[1]]] = round(num, 4)
        Matrix[ dicts[i[1]]][dicts[i[0]]] = round(num, 4)
        
    return Matrix