import numpy as np
import pandas as pd
from scipy import stats

class RegressionModel():
    def __init__(self, x, y, create_intercept = True, regression_type = "OLS"):
        self.x = x.copy()
        self.y = y.copy()
        self.create_intercept = create_intercept
        self.regression_type = regression_type
        self.results = {}

        if self.create_intercept:
            self.add_intercept()
    
    def add_intercept(self):
        if "intercept" not in self.x.columns:
            self.x = self.x.assign(intercept = pd.Series([1] * self.x.shape[0], index = self.x.index))

    def ols_regression(self):
        X = self.x.to_numpy(dtype=np.float64)
        y = self.y.to_numpy(dtype=np.float64).flatten()

        XTX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XTX_inv @ X.T @ y

        y_hat = (X @ beta_hat).flatten()
        residuals = y - y_hat

        n = X.shape[0]
        k = X.shape[1]

        sigma2 = float((residuals.T @ residuals) / (n - k))
        var_beta = sigma2 * XTX_inv
        se_beta = np.sqrt(np.diag(var_beta))

        t_stats = beta_hat / se_beta

        p_values = 1 - stats.t.cdf(t_stats, df=n - k)
        p_values = np.clip(p_values, 0, 1)

        self.results = {}
        for i, col in enumerate(self.x.columns):
            self.results[col] = {
                "coefficient": float(beta_hat[i]),
                "standard_error": float(se_beta[i]),
                "t_stat": float(t_stats[i]),
                "p_value": float(p_values[i])
            }
    
    def summary(self):

        summary_df = pd.DataFrame([
            {"Variable name": var,
             "coefficient value": res["coefficient"],
             "standard error": res["standard_error"],
             "t-statistic": res["t_stat"],
             "p-value": res["p_value"]
            }
            for var, res in self.results.items()
        ])

        print(summary_df.to_string(index = False))
    
        return summary_df

data = pd.read_csv("tests/files/assignment8Data.csv")
x = data[['sex','age','educ','white']]
y = data['incwage']
reg = RegressionModel(x, y, create_intercept=True)

w = reg.ols_regression()
print(reg.summary())