import numpy as np
import pandas as pd 
from lssvr import LSSVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from sko.PSO import PSO

# 1. read data

data_file = "data_382.csv"

data = pd.read_csv(data_file)

x = np.array(data.iloc[:383,1:10])
y = np.array(data.iloc[:383,10:11])

# 2. data normalization

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1) 

ss_x = MinMaxScaler() # x' = (x-x_min)/(x_max-x_min)
x_train = ss_x.fit_transform(x_train) 
x_test = ss_x.transform(x_test) 
ss_y = MinMaxScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = ss_y.transform(y_test.reshape(-1, 1)).ravel()

# 3. PSO

def lssvr_func(x):
	C, gamma = x
	model = LSSVR(kernel='rbf', gamma=gamma, C=C)
	model.fit(x_train, y_train)
	y_predict = model.predict(x_test)

	real_y_test = ss_y.inverse_transform(y_test.reshape(-1, 1))
	real_y_predict = ss_y.inverse_transform(y_predict.reshape(-1, 1))

	r2_s = r2_score(real_y_test, real_y_predict)
	rmse = mean_squared_error(real_y_test, real_y_predict)**0.5
	print('C =',C,'gamma =',gamma,'RMSE =',rmse,'R2 =',r2_s)
	return rmse

pso = PSO(func=lssvr_func, n_dim=2, pop=30, max_iter=200, lb=[1e-5, 1e-5], ub=[1e5, 1e5], w=0.8, c1=2, c2=2)

pso.run()

print('best_C_gamma = ', pso.gbest_x)
print('best_rmse = ', pso.gbest_y)


