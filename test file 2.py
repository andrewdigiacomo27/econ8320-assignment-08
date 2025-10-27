#si-exercise

import pandas as pd
import numpy as np
from scipy.stats import t

class RegressionModel(object):
    def __init__(self, x, y, create_intercept=True, regression_type="ols"):
        self.x = pd.DataFrame(x)
        self.y = pd.DataFrame(y)
        self.create_intercept = create_intercept
        self.regression_type = regression_type
        self.results = {}

        if self.create_intercept:
            if "intercept" not in self.x.columns:
                self.add_intercept()

    def add_intercept(self):
        self.x = x.assign(intercept=pd.Series([1]*np.shape(x)[0], index = self.x.index))
    def ols_regression(self):
        if self.regression_type == "ols":                                #key is variable
            X = np.array(self.x)
            Y = np.array(self.y).reshape(-1,1)
            n, k = X.shape                                               #df parameters
            df = n - k
            beta_coef = np.linalg.inv(X.T @ X) @ X.T @ Y               #coefficient
            s_2 = ((Y.T @ Y) - (Y.T @ X @ np.linalg.inv(X.T @ X) @ X.T @ Y)) / df #unbiased variance matrix
            cov_B = s_2 * np.linalg.inv(X.T @ X)                         #covariance of b
            variance = np.diag(cov_B)                                    #variance
            standardError = np.sqrt(variance)                            #sqrt of variance
            t_stat = beta_coef.flatten() / standardError.flatten()       #t stat
            p_val = t.sf(t_stat, df)                                     #p value

            for num, val in enumerate(self.x.columns):
                self.results[val] = {
                    "coefficient": float(beta_coef[num]),
                    "standard_error": float(standardError[num]),
                    "t_stat": float(t_stat[num]),
                    "p_value": float(p_val[num]),
                }

            return self.results
    def summary(self):
        if not self.results:
            self.ols_regression()

        var_name = []
        coef_value = []
        standardError = []
        t_stat = []
        p_value = []

        for key, value in self.results.items():
            var_name.append(key)
            coef_value.append(value['coefficient'])
            standardError.append(value['standard_error'])
            t_stat.append(value['t_stat'])
            p_value.append(value['p_value'])


        table_summary = pd.DataFrame({
            "Variable name": var_name,
            "coefficient value": coef_value,
            "standard error": standardError,
            "t-statistic": t_stat,
            "p-value": p_value
            })

        return table_summary
        
data = pd.read_csv("tests/files/assignment8Data.csv")
x = data[['sex','age','educ','white']]
y = data['incwage']
reg = RegressionModel(x, y, create_intercept=True)

w = reg.ols_regression()
print(reg.summary())

