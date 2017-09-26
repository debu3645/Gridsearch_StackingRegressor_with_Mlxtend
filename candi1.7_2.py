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

#X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y1, test_size = 0.20, random_state=1)


# Feature Scaling
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


# LINEAR REGRESSION MODEL FOR CPU
def sbg_linearmod(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_LinearReg_Memory.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    logging.info("tensor_bp sbg_linearmod iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data,  mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape


#STOCH. GRAD. DESC MODEL FOR CPU
def sbg_linearSGD(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = linear_model.SGDRegressor(penalty='l2', alpha=0.001, n_iter=1000)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename2 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_SGD_Memory.log'
    logging.basicConfig(filename=filename2, level=logging.DEBUG)
    logging.info("tensor_bp sbg_linearSGD iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# LASSO LINEAR REGRESSION MODEL FOR CPU
def sbg_lasso(iterate):
    iterate+=9000
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #reg = linear_model.Lasso(alpha=.3)
    reg = linear_model.Lasso(alpha = 1, max_iter = 3000,normalize = 'True', selection = 'random', tol = 0.001)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename3 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_Lasso_Memory.log'
    logging.basicConfig(filename=filename3, level=logging.DEBUG)
    logging.info("tensor_bp sbg_lasso iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# LASSO-LARS LINEAR REGRESSION MODEL FOR CPU
def sbg_lassolars(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = linear_model.LassoLars(alpha=.001)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename4 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_LassoLars_Memory.log'
    logging.basicConfig(filename=filename4, level=logging.DEBUG)
    logging.info("tensor_bp sbg_lassolars iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# BAYESIAN-RIDGE LINEAR REGRESSION MODEL FOR CPU
def sbg_bayesian(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename5 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_Bayesian_Memory.log'
    logging.basicConfig(filename=filename5, level=logging.DEBUG)
    logging.info("tensor_bp sbg_bayesian iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# GAUSSIAN NAIVE BAYES REGRESSION MODEL FOR CPU
def sbg_naivebayes(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = GaussianNB()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename6 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_NaiveBayes_Memory.log'
    logging.basicConfig(filename=filename6, level=logging.DEBUG)
    logging.info("tensor_bp sbg_naivebayes iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# ELASTIC NET REGRESSION MODEL FOR CPU
def sbg_elasticnet(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = ElasticNet(alpha=0.2, l1_ratio=1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename7 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_ElasticNet_Memory.log'
    logging.basicConfig(filename=filename7, level=logging.DEBUG)
    logging.info("tensor_bp sbg_elasticnet iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# RIDGE REGRESSION MODEL FOR CPU
def sbg_ridge(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = Ridge(alpha=0.01,tol=0.1,solver='sag')
    ## Use Recursive feature elimination to find out 30 most important elements.
    #selector = RFE(reg, 25, step=1)
    #selector = selector.fit(X_train, y_train)
    #print ("Supported Parameters: ",selector.support_)
    #print ("X columns: ", df_X.columns)
    #print ("Ranking: ", selector.ranking_)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    #selector = RFE(estimator, 5, step=1)

    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename8 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_Ridge_Memory.log'
    logging.basicConfig(filename=filename8, level=logging.DEBUG)
    logging.info("tensor_bp sbg_ridge iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# SUPPORT VECTOR REGRESSION MODEL FOR CPU
def sbg_supportvector(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename9 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_SupportVector_Memory.log'
    logging.basicConfig(filename=filename9, level=logging.DEBUG)
    logging.info("tensor_bp sbg_supportvector iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# POLYNOMIAL REGRESSION MODEL FOR CPU
def sbg_polynomial(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = PolynomialFeatures(degree=2)
    #x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)
    x_train_ = reg.fit_transform(x_train)
    x_test_ = reg.fit_transform(X_test)
    #y_train_ = reg.fit_transform(y_train)
    lg = LinearRegression()
    lg.fit(x_train_, np.ravel(y_train))
    y_pred = np.round(lg.predict(x_test_))
    y_test = y_test.values
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mean_squared_error(y_test, y_pred), rmse

# POLYNOMIAL PIPELINE MODEL FOR CPU
def sbg_polypipe(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = linear_model.ElasticNet(max_iter=2000,alpha=83,l1_ratio=0.1)
    reg = Pipeline([('poly', PolynomialFeatures(interaction_only=True)),('model', reg)])
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename10 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_PolyPipe_Memory.log'
    logging.basicConfig(filename=filename10, level=logging.DEBUG)
    logging.info("tensor_bp sbg_polypipe iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape

# RANDOM FOREST REGRESSOR FOR MEMORY
def sbg_randomforestreg(iterate):

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y2, test_size=0.20, random_state=iterate)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    reg = RandomForestRegressor(max_depth=5, random_state=iterate, n_estimators =100)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    dev_Memory = abs(y_pred - y_test.values)
    mean_dev = np.mean(dev_Memory)
    mse_Memory = np.sqrt(np.sum(dev_Memory ** 2) / dev_Memory.size)
    mape = np.mean(dev_Memory / y_test.values)
    max_pe = np.max(dev_Memory)
    max_ne = np.max(np.negative(dev_Memory))

    new_data1 = pd.DataFrame(y_pred)
    new_data2 = pd.DataFrame(y_test.values)
    new_data = pd.merge(new_data1, new_data2, left_index=True, right_index=True)

    filename10 = r'C:\Users\epatdeb\AlphaCANDI\SBG_Rawinput_1.6\latest\Logs\AlphaCandi17_PolyPipe_Memory.log'
    logging.basicConfig(filename=filename10, level=logging.DEBUG)
    logging.info("tensor_bp sbg_randomforestreg iter:%s \n \n y_pred/y_test: \n %s \n mae:%s mse:%s mape:%s max_pe:%s max_ne:%s",
                 iterate, new_data, mean_dev, mse_Memory, mape, max_pe, max_ne)
    logging.shutdown()

    return mean_squared_error(y_test, y_pred), mean_dev, mape


round = 100

mse_linreg = []
mae_linreg = []
mape_linreg = []

mse_linSGD = []
mae_linSGD = []
mape_linSGD = []

mse_lasso = []
mae_lasso = []
mape_lasso = []

mse_laslars = []
mae_laslars = []
mape_laslars = []

mse_naivebayes = []
mae_naivebayes = []
mape_naivebayes = []

mse_bayesian = []
mae_bayesian = []
mape_bayesian = []

mse_elastic = []
mae_elastic = []
mape_elastic = []

mse_ridge = []
mae_ridge = []
mape_ridge = []

mse_supportvector = []
mae_supportvector = []
mape_supportvector = []

mse_polypipe = []
mae_polypipe = []
mape_polypipe = []

mse_randfreg = []
mae_randfreg = []
mape_randfreg = []

for i in range (round):

    #print ("Started the test for total {} rounds. Current round is: {} ".format(round, i))

    x, y, z = sbg_linearmod(i)
    x2, y2, z2 = sbg_linearSGD(i)
    x3, y3, z3 = sbg_lasso(i)
    x4, y4, z4 = sbg_lassolars(i)
    x5, y5, z5 = sbg_ridge(i)
    x6, y6, z6 = sbg_elasticnet(i)
    #x7, y7, z7 = sbg_naivebayes(i)
    x8, y8, z8 = sbg_bayesian(i)
    x9, y9, z9 = sbg_supportvector(i)
    x10, y10, z10 = sbg_polypipe(i)
    x11, y11, z11 = sbg_randomforestreg(i)


    mse_linreg.append(x)
    mae_linreg.append(y)
    mape_linreg.append(z)

    mse_linSGD.append(x2)
    mae_linSGD.append(y2)
    mape_linSGD.append(z2)

    mse_lasso.append(x3)
    mae_lasso .append(y3)
    mape_lasso.append(z3)

    mse_laslars.append(x4)
    mae_laslars.append(y4)
    mape_laslars.append(z4)

    mse_ridge.append(x5)
    mae_ridge.append(y5)
    mape_ridge.append(z5)

    mse_elastic.append(x6)
    mae_elastic.append(y6)
    mape_elastic.append(z6)

    #mse_naivebayes.append(x7)
    #mae_naivebayes.append(y7)
    #mape_naivebayes.append(z7)

    mse_bayesian.append(x8)
    mae_bayesian.append(y8)
    mape_bayesian.append(z8)

    mse_supportvector.append(x9)
    mae_supportvector.append(y9)
    mape_supportvector.append(z9)

    mse_polypipe.append(x10)
    mae_polypipe.append(y10)
    mape_polypipe.append(z10)

    mse_randfreg.append(x11)
    mae_randfreg.append(y11)
    mape_randfreg.append(z11)

print ("Average Linear MSE: ", sum(mse_linreg)/round)
print ("Average Linear MAE: ", sum(mae_linreg)/round)
print ("Average Linear MAPE: ", sum(mape_linreg)/round)

print ("Average SGD MSE: ", sum(mse_linSGD)/round)
print ("Average SGD MAE: ", sum(mae_linSGD)/round)
print ("Average SGD MAPE: ", sum(mape_linSGD)/round)

print ("Average Lasso MSE: ", sum(mse_lasso)/round)
print ("Average Lasso MAE: ", sum(mae_lasso)/round)
print ("Average Lasso MAPE: ", sum(mape_lasso)/round)

print ("Average LassoLars MSE: ", sum(mse_laslars)/round)
print ("Average LassoLars MAE: ", sum(mae_laslars)/round)
print ("Average LassoLars MAPE: ", sum(mape_laslars)/round)

#print ("Average NaivBayes MSE: ", sum(mse_naivebayes)/round)
#print ("Average NaivBayes MAE: ", sum(mae_naivebayes)/round)
#print ("Average NaivBayes MAPE: ", sum(mape_naivebayes)/round)

print ("Average Bayesian MSE: ", sum(mse_bayesian)/round)
print ("Average Bayesian MAE: ", sum(mae_bayesian)/round)
print ("Average Bayesian MAPE: ", sum(mape_bayesian)/round)

print ("Average ElasticNet MSE: ", sum(mse_elastic)/round)
print ("Average ElasticNet MAE: ", sum(mae_elastic)/round)
print ("Average ElasticNet MAPE: ", sum(mape_elastic)/round)

print ("Average Ridge MSE: ", sum(mse_ridge)/round)
print ("Average Ridge MAE: ", sum(mae_ridge)/round)
print ("Average Ridge MAPE: ", sum(mape_ridge)/round)

print ("Average SVR MSE: ", sum(mse_supportvector)/round)
print ("Average SVR MAE: ", sum(mae_supportvector)/round)
print ("Average SVR MAPE: ", sum(mape_supportvector)/round)

print ("Average PolyPipe MSE: ", sum(mse_polypipe)/round)
print ("Average PolyPipe MAE: ", sum(mae_polypipe)/round)
print ("Average PolyPipe MAPE:", sum(mape_polypipe)/round)

print ("Average RndmFrstReg MSE: ", sum(mse_randfreg)/round)
print ("Average RndmFrstReg MAE: ", sum(mae_randfreg)/round)
print ("Average RndmFrstReg MAPE:", sum(mape_randfreg)/round)

