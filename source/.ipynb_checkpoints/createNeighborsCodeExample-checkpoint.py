# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 10:34:08 2022

@author: owner
"""

import numpy as np
test = np.random.rand(365*7*48,8)
# Pad with nan
test1 = np.pad(test, ((49,49),(0,0)),mode='constant',constant_values = (np.nan,np.nan))
test_select = np.concatenate((test1[49:-49,:],test1[0:-98,:],
                              test1[98:,:],test1[48:-50,:],
                             test1[50:-48,:]),axis=1)