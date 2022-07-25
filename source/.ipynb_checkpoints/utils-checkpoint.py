import itertools
import pandas as pd

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

###########################################

def creat_lags (df, cols, arr_shifts = [ 0, 1, -1, 49, -49]):
     # https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/   
    shifted_cols = []
    shifted_names = []
    
    for col in cols:
        df_col = df.loc[:,[col]]
        # print(df_col.head (6))
        for sh in arr_shifts:
            shifted_cols.append(df_col.shift(sh))
            shifted_names.append(col+"_"+str(sh))
    
    dataframe = pd.concat(shifted_cols, axis=1)
    dataframe.columns = shifted_names
    
    # print(dataframe.head(5))
    return dataframe