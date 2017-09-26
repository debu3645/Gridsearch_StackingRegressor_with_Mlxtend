# Gridsearch_StackingRegressor_with_Mlxtend
# Creating  a stacking model for a Regression problem using SKlearn and Mlxtend.


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

