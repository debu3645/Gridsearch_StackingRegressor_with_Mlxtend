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
import logging
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.feature_selection import RFE
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

#clms_Y1 = [u'CPU (Erlang Scheduler, %)']
clms_Y2 = [u'Memory (Erlang Beam, MB)']
df1.columns = clms
df_X = df1[clms_X]
#df_Y1 = df1[clms_Y1]
df_Y2 = df1[clms_Y2]

splitclm = [u'vCPU No.', u'No. of BGF', u'LB (eVIP)', u'T1 Timer (s)', u'Timer B (s)',u'SMM', u'REGed Users (K)',
            u'Gm over UDP (%)',u'Gm over TCP (%)',  u'Gm over TLS (%)',u'ADM via Rx (%)', u'ACC via Rf (%)',
            u'IPv4 (%)', u'IPv6 (%)', u'VoLTE call rate',u'Non-VoLTE call rate', u'Avg. answer time (s)',
            u'Avg. holding time (s)', u'IMS AKA (%)', u'Access re-reg interval (m)',u'Init-reg  call rate',
            u'Access re-reg interval (m)', u'Core re-reg interval (m)', u'Subs call rate',
            u'Con. Subs Dialogs (K)', u'Stateless Subs dialogs (%)', u'Publish call rate', u'Notify call rate',
            u'Message call rate',u'Options call rate']

df1_X = df1[splitclm]

sc = StandardScaler()

def sbg_mlxtend_ensamble(iterate):
    iterate+=501
    lin_mod = linear_model.LinearRegression()
    bsn_rdg = linear_model.BayesianRidge()
    elstc_nt = ElasticNet(alpha=0.2, l1_ratio=1)
    ridge = Ridge(alpha=0.01, tol=0.1, solver='sag')
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    sgd_reg = linear_model.SGDRegressor(penalty='l2', alpha=0.001, n_iter=1000)
    lasso_reg = linear_model.Lasso(alpha = 1, max_iter = 3000,normalize = 'True', selection = 'random', tol = 0.001)
    rndm_frst = RandomForestRegressor(max_depth=5, n_estimators=10)

    stregr = StackingRegressor(regressors=[ sgd_reg, rndm_frst ],
                           meta_regressor= ridge)

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    stregr.fit(X_train, y_train)
    y_pred = stregr.predict(X_test)

    #print("Mean Squared Error: %.4f"
    #      % np.mean((y_pred - y_test.values) ** 2))
    #print('Variance Score: %.4f' % stregr.score(X_test, y_test.values))

    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))
    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename12 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_MlxEnsmbl_Memory.log'
    logging.basicConfig(filename=filename12, level=logging.DEBUG)
    logging.info(
        "tensor_bp sbg_mlxtend_ensamble iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
        iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

mse_mlxensmbl = []
mae_mlxensmbl = []
mape_mlxensmbl = []

round = 100

for i in range (round):

    print ("Started the test for total {} rounds. Current round is: {} ".format(round, i))

    x12, y12, z12 = sbg_mlxtend_ensamble(i)

    mse_mlxensmbl.append(x12)
    mae_mlxensmbl.append(y12)
    mape_mlxensmbl.append(z12)

print ("Average MlxEnsambl MSE: ", sum(mse_mlxensmbl)/round)
print ("Average MlxEnsambl MAE: ", sum(mae_mlxensmbl)/round)
print ("Average MlxEnsambl MAPE:", sum(mape_mlxensmbl)/round)