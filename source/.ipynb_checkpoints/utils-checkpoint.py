import itertools

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
    var_comb = list(itertools.combinations(cols, 2))
    for i in cols:
        var_comb.append((i,i))
    return var_comb