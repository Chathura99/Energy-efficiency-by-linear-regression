import numpy as np
import pandas as pd
# plot
import matplotlib.pyplot as plt
import seaborn as sns
# train model
from sklearn.model_selection import train_test_split
# Import the os module
import os

# Get the current working directory
cwd = os.getcwd()
print("path",cwd)

energy_data=pd.read_csv('D:\\Chathura\\UGVLe\\3Y1S\\SCS_3201 ML 2\\3\\19001606\\DATA SCIENCE\Energy efficiency')
# X1 Relative Compactness
# X2 Surface Area
# X3 Wall Area
# X4 Roof Area
# X5 Overall Height
# X6 Orientation
# X7 Glazing Area
# X8 Glazing Area Distribution
# y1 Heating Load
# y2 Cooling Load

energy_data.columns=["relative_compactness","surface_area","wall_area","roof_area","overall_height","orientaion",
                   "glazing_area","glazing_area_dist","heating_load","cooling_load"]

energy_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(energy_data.corr(), annot=True)

energy_data.corr()['cooling_load']
energy_data.corr()['heating_load']

new_energy_data = energy_data.drop(['relative_compactness','roof_area','orientaion','glazing_area_dist'], axis=1)


X=new_energy_data.drop(['heating_load','cooling_load'], axis=1)
Y=new_energy_data[['heating_load','cooling_load']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

def model_acc(model):
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(str(model)+ ' --> ' +str(acc))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

findload1=lr.fit(x_train,y_train)

print("Score :",findload1.score(x_test, y_test))

pred1 = findload1.predict(x_test)