# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:50:40 2020

@author: YJ001
"""

import pandas as pd
data = pd.DataFrame(data = np.hstack([x_, y_]), columns = ["x", "y"])
print(data.head())

import statsmodels.formula.api as smf
mod = smf.quantreg('y ~ x', data)
res = mod.fit(q=.5)
print(res.summary())