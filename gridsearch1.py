from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold, KFold, cross_val_score

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPRegressor


df = pd.read_excel('v16_test_data.xlsx')
df.drop('CASE NAME', axis =1, inplace=True)
df1 = df.iloc[1:]

clms = [u'Hypervisor-KVM', u'HW Type (GEP7/GEP5)', u'vCPU No.', u'Freq (GHz)',
 u'Hyperthreading', u'CPU Pinning', u'X-sockets', u'Power Saving', u'vCPU for DPDK',
 u'No. of PCRF', u'No. of BGF', u'LB (eVIP)', u'T1 Timer (s)', u'Timer B (s)', u'SMM',
 u'REGed Users (K)', u'Gm over UDP (%)', u'Gm over TCP (%)', u'Gm over TLS (%)',
 u'ADM via Rx (%)', u'ACC via Rf (%)', u'IPv4 (%)', u'IPv6 (%)', u'VoLTE call rate',
 u'Non-VoLTE call rate', u'Avg. answer time (s)', u'Avg. holding time (s)',
 u'Init-reg  call rate', u'IMS AKA (%)', u'Access re-reg interval (m)',
 u'Core re-reg interval (m)', u'De-reg call rate', u'Subs call rate',
 u'Con. Subs Dialogs (K)', u'Stateless Subs dialogs (%)', u'Publish call rate',
 u'Notify call rate', u'Message call rate', u'Options call rate',
 u'CPU (vmstate, %)', u'CPU (Erlang Scheduler, %)', u'Memory (System, MB)',
 u'Memory (Erlang Beam, MB)']

clms_X = [u'Hypervisor-KVM', u'HW Type (GEP7/GEP5)', u'vCPU No.', u'Freq (GHz)',
 u'Hyperthreading', u'CPU Pinning', u'X-sockets', u'Power Saving', u'vCPU for DPDK',
 u'No. of PCRF', u'No. of BGF', u'LB (eVIP)', u'T1 Timer (s)', u'Timer B (s)', u'SMM',
 u'REGed Users (K)', u'Gm over UDP (%)', u'Gm over TCP (%)', u'Gm over TLS (%)',
 u'ADM via Rx (%)', u'ACC via Rf (%)', u'IPv4 (%)', u'IPv6 (%)', u'VoLTE call rate',
 u'Non-VoLTE call rate', u'Avg. answer time (s)', u'Avg. holding time (s)',
 u'Init-reg  call rate', u'IMS AKA (%)', u'Access re-reg interval (m)',
 u'Core re-reg interval (m)', u'De-reg call rate', u'Subs call rate',
 u'Con. Subs Dialogs (K)', u'Stateless Subs dialogs (%)', u'Publish call rate',
 u'Notify call rate', u'Message call rate', u'Options call rate' ]

clms_Y2 = [u'Memory (Erlang Beam, MB)']
df1.columns = clms
df_X = df1[clms_X]
df_Y2 = df1[clms_Y2]

sc = StandardScaler()
df_X_sc = sc.fit_transform(df_X)

#reg = linear_model.LinearRegression()
reg_sgd = linear_model.SGDRegressor()
pnlty = ['l1', 'l2','elasticnet']
power = [0.1, 0.3, 0.5, .9, 1, .01, .03, 0.06,0.09, 0.001,0.003,0.006,0.009]
#lpha = [0.0001, 0.0003, 0.0006, 0.0009, 0.001, 0.003,0.006,0.009,0.01, 0.03,0.06,0.09,0.1,0.3,0.6,0.9]
eta = [0.1, 0.3, 0.5, .9, 1, .01, .03, 0.06,0.09, 0.001,0.003,0.006,0.009]
iters = [500,1000]



grid_params = dict(penalty = pnlty, power_t = power, n_iter=iters, eta0=eta )


Grids = GridSearchCV( reg_sgd,grid_params,scoring='neg_mean_squared_error',cv=10)
print ("****"*10)
print ("****"*10)
print ("****"*10)
print("STarted....")
Grids.fit(df_X, df_Y2)
print ("Grids.best_estimator_: ", Grids.best_estimator_)
print ("****"*10)
print ("Best Parameters: ", Grids.best_params_)
print ("****"*10)
print ("Best Scores : ", Grids.best_score_)
print ("****"*10)

print("Finished....")