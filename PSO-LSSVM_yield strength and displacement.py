import numpy as np
import pandas as pd 
from lssvr import LSSVR
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler

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

# 3. LSSVM

gamma = 2.3036
C = 195.453

model = LSSVR(kernel='rbf', gamma=gamma, C=C) 

model.fit(x_train, y_train)

# 4. Output

outTestData = 'Testing Result for yield strength.csv'

y_predict = model.predict(x_test)

real_y_test = ss_y.inverse_transform(y_test.reshape(-1, 1))
real_y_predict = ss_y.inverse_transform(y_predict.reshape(-1, 1))

r2_s = r2_score(real_y_test, real_y_predict)
rmse = mean_squared_error(real_y_test, real_y_predict)**0.5
mae = mean_absolute_error(real_y_test, real_y_predict)
ev = explained_variance_score(real_y_test, real_y_predict)

with open(outTestData, 'w') as f:
	f.write("R2,"+str(r2_s)+"\n")
	f.write("rmse,"+str(rmse)+"\n")
	f.write("mae,"+str(mae)+"\n")
	f.write("ev,"+str(ev)+"\n")
	f.write("predict,identify\n")
	for i in range(len(real_y_test)):
		f.write(str(real_y_predict[i][0])+","+str(real_y_test[i][0])+"\n")

outTrainData = 'Training Result for yield strength.csv'

y_predict_train = model.predict(x_train)

real_y_train = ss_y.inverse_transform(y_train.reshape(-1, 1))
real_y_predict_train = ss_y.inverse_transform(y_predict_train.reshape(-1, 1))

r2_s_train = r2_score(real_y_train, real_y_predict_train)
rmse_train = mean_squared_error(real_y_train, real_y_predict_train)**0.5
mae_train = mean_absolute_error(real_y_train, real_y_predict_train)
ev_train = explained_variance_score(real_y_train, real_y_predict_train)

with open(outTrainData, 'w') as f_train:
	f_train.write("r2_s_train,"+str(r2_s_train)+"\n")
	f_train.write("rmse_train,"+str(rmse_train)+"\n")
	f_train.write("mae_train,"+str(mae_train)+"\n")
	f_train.write("ev_train,"+str(ev_train)+"\n")
	f_train.write("predict,identify\n")
	for i in range(len(real_y_predict_train)):
		f_train.write(str(real_y_predict_train[i][0])+","+str(real_y_train[i][0])+"\n")

# 5. Draw

def draw_R(identify ,predict, r2_s, para):
	plt.figure()
	plt.xlabel('Predicted'+para, fontsize=16)
	plt.ylabel('Identified'+para, fontsize=16)
	axle_max = max([max(identify), max(predict)])*(1.1)
	axle_min = min([min(identify), min(predict)])*(0.9)
	plt.xlim(axle_min, axle_max)
	plt.ylim(axle_min, axle_max)
	plt.plot([axle_min, axle_max], [axle_min, axle_max], c='black')
	plt.scatter(predict, identify, c='none', edgecolor='r', marker='o', s=30)
	
	plt.title('R2 = %f' % r2_s)
	plt.show()


draw_R(real_y_test ,real_y_predict, r2_s, 'yield strength')

draw_R(real_y_train ,real_y_predict_train, r2_s_train, 'yield strength')







