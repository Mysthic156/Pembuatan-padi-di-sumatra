# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="qAvCzo5o5bJX"
# # **Prediction of Rice Production on Sumatera Island, Indonesia**

# + [markdown] id="Pv-l-u166QQ_"
# Sumatera Island has more than 50 percent of agricultural land in each province with the most dominant main food commodity is rice, while other minor commodities are corn, peanuts and sweet potatoes. Agricultural produce in Sumatera Island is highly vulnerable to climate change and its negative impacts can affect cropping patterns, planting time, production and yield quality. Moreover, the increase in the earth's temperature due to the impact of global warming which will affect the pattern of precipitation, evaporation, water-run off, soil moisture, and climate variations which are very fluctuating as a whole can threaten the success of agricultural production.
#
# The data is related information records from 1993 to 2020 covering 8 provinces on Sumatera Island, namely Nanggroe Aceh Darussalam, North Sumatera, West Sumatera, Riau, Jambi, South Sumatera, Bengkulu and Lampung.

# + [markdown] id="m9juxv_r83Wc"
# ## Goal of this notebook: 
# The goal of this notebook is to predict the rice production on Sumatera Island based on data accumulated over the previous 28 years.

# + [markdown] id="MURe3XhoA1Kp"
# # 1. Load Dataset

# + id="BK-PvjjR5HhM"
#import library package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="1cwF_EUCBgoI" outputId="44ed7a27-ebb1-41d7-c999-aabfe09beb90"
data = pd.read_csv('Data_Tanaman_Padi_Sumatera_version_1.csv')
data

# + colab={"base_uri": "https://localhost:8080/"} id="ZnUab0uYHYcu" outputId="7ff5a857-5700-46e9-bf5a-4f2db3b54973"
#check row and column
data.shape

# + [markdown] id="c-42bd-NDciF"
# The total data is 224 which for each province has 28 annual data.
#
# The dataset consists of the following attributes:
# 1. **Province**: Name of province
# 2. **Year**: Year of rice production
# 3. **Production**: Production results or annual harvest (tons)
# 4. **Land Area**: Agricultural area (hectares)
# 5. **Rainfall**: Average amount of rainfall in a year (millimeters)
# 6. **Humidity**: Average humidity level in a year (percentage)
# 7. **Average Temperature**: The average degree of temperature in a year (celsius)
#
# Attributes number 1 - 4 collected from the Indonesian Central Bureau of Statistics Database, and other attributes are collected from the Indonesian Agency for Meteorology, Climatology and Geophysics Database

# + colab={"base_uri": "https://localhost:8080/"} id="w4PUOdXiQIkZ" outputId="1a4d2b3f-323f-4104-ac31-09e69f3d5844"
data.info()

# + [markdown] id="fcAjPV3XNChN"
# # 2. Exploratory Data Analysis

# + colab={"base_uri": "https://localhost:8080/"} id="_TLUvtU5QeTy" outputId="82cd0863-25c9-4279-d255-39e4329091c0"
data.isnull().sum()

# + colab={"base_uri": "https://localhost:8080/", "height": 300} id="7tzPa--OBoud" outputId="8429cfa9-359d-4f78-b477-02d1df139422"
#overview statistics descriptive
data.describe()

# + [markdown] id="WuOBL7yDIZL6"
# The average yield in 8 provinces for 28 years was 1679700.887 tons with the lowest yield was 42938 tons and the highest was 4881089 tons. The average area of ​​agricultural land is 374350 hectares.
#
# From the data description above, it can be seen that the mean and median values ​​of each attribute are not much different. So it can be said that the data is normally distributed.
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 563} id="eBcmxLd6JXFj" outputId="5d893ad2-766e-4b4d-e3e8-eb4f2655c5a2"
plt.figure(figsize=(15,8))
sns.barplot(x='Provinsi', y='Produksi', data=data)
plt.show()

# + [markdown] id="iRquLIl-KGr5"
# North Sumatera has the highest yield production in the dataset.

# + colab={"base_uri": "https://localhost:8080/", "height": 332} id="BXUoMXCiwgc0" outputId="459c2f90-c2f5-45c4-ced8-9861f3211c10"
pivot_table = pd.pivot_table(data, 
                             index='Provinsi', 
                             values='Produksi', 
                             aggfunc=np.mean)

# + colab={"base_uri": "https://localhost:8080/", "height": 578} id="dofVRFUpJpJJ" outputId="d57999e3-c2b3-4718-d6fa-9c32e9bd06e8"

plt.figure(figsize=(15,8))

sns.barplot(x='Tahun', y='Produksi', data=data)

plt.xticks(rotation=45)

plt.show()

# + [markdown] id="EyTVTnbOLnqT"
# The year with the largest production was in 2017, but in the following years production has decreased significantly.

# + colab={"base_uri": "https://localhost:8080/", "height": 563} id="JkMrGM4hLE88" outputId="a48d98b6-5bda-438e-eb9b-be1ddb1693e4"
plt.figure(figsize=(30,8))

sns.lineplot(x='Suhu rata-rata', y='Produksi', data=data)
plt.show()

# + [markdown] id="Jz4P0lmKMgb4"
# Maximum production occurs when the average temperature is in the range of 27-28 degrees Celsius.

# + colab={"base_uri": "https://localhost:8080/", "height": 552} id="qxVWm_leR2dp" outputId="baf5f87c-6f5d-4fb0-a97a-fa052e5522bd"
plt.figure(figsize=(30,8))
sns.lineplot(x='Tahun', y='Suhu rata-rata', data=data, color='red')
plt.show()

# + [markdown] id="Cyt0AZvfTEbo"
# The highest average temperature was ever achieved in the year 2000-2005, which was more than 29 degrees Celsius, but in the following years it tended to be stable in the temperature range of 27 degrees Celsius.

# + colab={"base_uri": "https://localhost:8080/", "height": 563} id="WZKJbyG6MjvC" outputId="8635c26e-7cd4-4fb2-c755-cf9ad518000c"

plt.figure(figsize=(30,8))
sns.lineplot(x='Curah hujan', y='Produksi', data=data, color='green')
plt.show()

# + [markdown] id="A9pegdRWOITD"
# It can be said that production may be more stable in moderate rainfall, which is around 2500mm per year.

# + colab={"base_uri": "https://localhost:8080/", "height": 563} id="PtZdJrT5ON6Q" outputId="4b229769-dd69-46c5-a022-74030311349d"
plt.figure(figsize=(30, 8))
sns.lineplot(x='Kelembapan', y='Produksi', data=data, color='purple')
plt.show()

# + [markdown] id="-eeaR3lNOxl8"
# At a humidity level of around 80-85%, production looks more optimal and stable.

# + [markdown] id="tXWBlJxlOd5G"
# Based on the purpose of this notebook, the attribute that will be the output is the harvest (column = Production).

# + [markdown] id="xHCohQSUOhTc"
# **a. Distribution of production quantities**

# + id="1T8K8Wl7NIHn"
#statistical plot
import scipy.stats as stats
import pylab as py

# + colab={"base_uri": "https://localhost:8080/", "height": 295} id="6jBc5mSTO-kD" outputId="92ec60d1-5d50-4d43-f87a-249acc6237fe"
#QQ-plot (Quantile-Quantile Plot)
stats.probplot(data['Produksi'], dist='norm', plot=py)
py.show()

# + [markdown] id="wAVnKqW6RbQp"
# The QQ-plot above shows how the data is distributed from the selected variable, namely **Production** variable. The plot above is a plot of the theoretical quantile values ​​with the quantile values ​​of the selected variables. So, the plot will describe how many values ​​in a distribution are above or below a certain limit. From the plot results above, it is obtained that the lines tend to be straight, which means that over all the data distribution of these variables tends to be normal.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="qxB-UCfkP86P" outputId="6a892c4c-438d-4d0c-f3b9-2855effb2db3"
sns.pairplot(data)

# + [markdown] id="6Q2B5rSQRxw7"
# **b. Checking the Outlier**

# + colab={"base_uri": "https://localhost:8080/", "height": 619} id="Y-f5zWDjPeAB" outputId="d2b6ae99-38ac-4e31-ac4d-8d509fe813b2"
data[['Produksi']].boxplot(figsize=(15,10))

# + [markdown] id="crZxjBx1SNpZ"
# From the results of the vertical boxplot visualization above, it can be seen that there are no outliers from the output variable, **Production**.

# + [markdown] id="yoicPyFpZVEN"
# **c. Correlation Matrix between numerical variables**

# + colab={"base_uri": "https://localhost:8080/"} id="QSKyI1Jea3nB" outputId="36202731-f8f7-4d9f-bd80-6619bd219ec2"
data.columns

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="m4lkLvJUcEbg" outputId="0875411d-589b-4fe5-cbed-fe86dc6625b4"
#change the column order
cols = list(data.columns)

df = data.copy()
df = df[cols[0:2] + cols[3:7] + [cols[2]]]
df.tail()

# + [markdown] id="_3Il56pIZpLQ"
# Note:
# - Correlation >= 0.5 there is a coherent relationship between variables
# - Correlation < 0.5 there is no close relationship between variables

# + colab={"base_uri": "https://localhost:8080/", "height": 238} id="2vDc6HXagNs3" outputId="c354705c-7f06-45da-89fb-cf50994e8188"


# + colab={"base_uri": "https://localhost:8080/", "height": 550} id="dVCBmDN0R4oR" outputId="8842d11a-0b5a-4d19-9741-70d89457e94b"
fig, ax = plt.subplots(figsize=(15,8), dpi=80)
sns.heatmap(df.loc[:, 'Luas Panen':'Produksi'].corr(), cmap='rocket', annot=True)
plt.title('Correlation Matrix Between Variables')
plt.show()

# + [markdown] id="F_TemhrrmQ-3"
# From the correlation matrix, it can be seen the type of correlation between the independent variables and the output variable. 
#
# The area of ​​agricultural land and the average temperature have a positive correlation value, which means that when the value of those independent variable is greater, the production will also increase, although it may not be significant (effect from temperature). 
#
# Meanwhile, the rainfall and humidity variables have a negative correlation value, which means that when those two variables get smaller, the production will increase (the movement of the graph moves in reverse), even though the relationship is not that strong.

# + [markdown] id="M1rowyfigkYs"
# # 3. Data Preprocessing (Feature Engineering)

# + colab={"base_uri": "https://localhost:8080/", "height": 550} id="dsD-3D2ww0Ot" outputId="87c1649a-38ca-42c0-e2ae-167a3562214f"
#checking multicollinearity
fig, ax = plt.subplots(figsize=(15,8), dpi=80)
sns.heatmap(df.loc[:, 'Luas Panen':'Suhu rata-rata'].corr(), cmap='rocket', annot=True)
plt.title('Correlation Matrix Between Independent Variables')
plt.show()

# + [markdown] id="kpUxTwEgrdJ9"
# **Multicollinearit**y is a situation that indicates a strong correlation or relationship between two or more independent variables in a regression model which will affect the stability and accuracy of the model. One way to detect the existence of multicollinearity in the regression model is by looking at the strength of the correlation between the independent variables. If there is a correlation between the independent variables > 0.8 it can indicate the presence of multicollinearity.
#
# So, from the matrix above it can be concluded that there is no multicollinearity found.

# + [markdown] id="Vw9Dp-wQuvHc"
# ## Feature Encoding

# + [markdown] id="urcP68e-u07g"
# Categorical data are variables that contain label values rather than numeric values. The number of possible values is often limited to a fixed set, like in this dataset, 'Province' values. Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric. This means that categorical data must be converted to a numerical form. 
#
# One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. For that purpose, One-Hot Encoding will be used to convert 'Province' columns to one-hot numeric array.
#
# The categorical value represents the numerical value of the entry in the dataset. This encoding will create a binary column for each category and returns a matrix with the results.

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="W34ylVpKjShZ" outputId="c01f0665-6da5-46f7-a0c8-fed267c9404c"
#one hot encoding
from sklearn.preprocessing import OneHotEncoder
string_feat = ['Provinsi']
ohe = OneHotEncoder()
ohe.fit(df[string_feat])
data_ohe_res = pd.DataFrame(ohe.transform(df[string_feat]).toarray(),
                          columns=ohe.get_feature_names_out())
df = pd.concat([df, data_ohe_res], axis=1)
df = df.drop(columns=string_feat)
df.head()

# + [markdown] id="kniDRJulKvCk"
# ## Feature Selection

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="LcFw3v88vo80" outputId="150fb7e2-2291-4638-e577-22e4fd6eb7d3"
df = df.drop(columns='Tahun')
df.tail()

# + id="kAw432QvLTeD"
x = df.drop('Produksi',axis=1)
y = df[['Produksi']]

# + [markdown] id="wPPfhffCLrLE"
# ## Train and Test Split

# + [markdown] id="wa14VkzBsHZQ"
# The dataset will be split to two datasets, the training dataset and test dataset. The data is usually tend to be split inequality because training the model usually requires as much data-points as possible.The common splits are 70/30 or 80/20 for train/test.
#
# The training dataset is the intial dataset used to train ML algorithm to learn and produce right predictions. (70% of dataset is training dataset).
#
# The test dataset, however, is used to assess how well ML algorithm is trained with the training dataset. We can’t simply reuse the training dataset in the testing stage because ML algorithm will already “know” the expected output, which defeats the purpose of testing the algorithm. (30% of dataset is testing dataset).

# + colab={"base_uri": "https://localhost:8080/"} id="p3GxSBV0L2fM" outputId="33809bd1-b85e-4911-8e5a-93a9bd5c6781"
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)
print('x_train :',x_train.shape)
print('x_test :',x_test.shape)
print('y_train :',y_train.shape)
print('y_test :',y_test.shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="w01R3jjUL8iL" outputId="5b1a3dcc-22ad-491c-d1ec-6e5cc87abd37"
x_train[:5]

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="zRTzA4s4rM2L" outputId="dab6d5cf-a7e4-4a5e-ebf3-2b9e8e6061cd"
y_test.tail()

# + [markdown] id="EgbbAGuAMfUg"
# ## Feature Scaling

# + [markdown] id="KOa3eZEPuQO7"
# The dataset contains features highly varying in magnitudes, units and range. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.
#
# To supress this effect, we need to bring all features to the same level of magnitudes. This can be acheived by scaling.

# + id="tk1Kpp0LMSmc"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

# + colab={"base_uri": "https://localhost:8080/"} id="SiPEsvNcNgYl" outputId="758fed9c-dfd2-4cee-b96b-6be74ad00720"
x_train[:5]

# + colab={"base_uri": "https://localhost:8080/"} id="3QyHp6ObNiv1" outputId="fc80b099-ebe9-46a3-be37-b1e265a3ea85"
np.set_printoptions(suppress=True)
print(x_train[:5])
print(y_train[:5])

# + colab={"base_uri": "https://localhost:8080/"} id="qC-Ft1ahS0ca" outputId="8f9d3a30-09f6-4998-869a-e631be143e54"
print(x_train.min())
print(x_train.max())

# + colab={"base_uri": "https://localhost:8080/"} id="iwPCxQ0GTwwG" outputId="1f23b071-cd27-448f-ae4a-beeaa0ddf708"
print(y_train.min())
print(y_train.max())

# + colab={"base_uri": "https://localhost:8080/"} id="G1jvrnMGqGRA" outputId="2295cf7c-6b0b-4cf1-e855-c8e7ff16799e"
print(x_test[:5])
print(y_test[:5])

# + [markdown] id="3oonpiPIOM_0"
# # 4. Modeling

# + [markdown] id="8LtS31TmybYO"
# Modeling is done using 6 algorithms namely: 
# 1. Linear Regression
# 2. Random Forest Regressor
# 3. Gradient Boosting
# 4. Support Vector Regressor 
# 5. Decision Tree Regressor and
# 6. K-Nearest Neighbors Regressor. 
#
# For each model will be applied hyperparameter tuning to increase model perform based on R2-score. To find the best parameters will use Grid Search CV or Randomized Search CV, and it will depends how the algorithm work.
#
# Cross-validation (CV) is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation.
#
# To check how the model predict, will represent visualization in the form of graphical images of distplots from the results estimated by the algorithm with the original data.
#
# Later an evaluation will be carried out to determine the best algorithm out of the six algorithms to be selected.
#
# **Note**: When finished doing hyperparameter tuning using Randomized Search CV, the code is used as a comment because when the code is run again, the tuning results will be different, even though the model scores are not much different.

# + id="NlrqusogN9Az"
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# + [markdown] id="El2UPHGhVPSg"
# ## Linear Regression

# + [markdown] id="OnqIw-yB0oX0"
# Linear Regression is an algorithm for regression modeling that is used to predict variable values based on the values of other variables.

# + colab={"base_uri": "https://localhost:8080/"} id="ppMZZ42K5zjF" outputId="139c41ec-04b2-43fb-e980-5660e32bcc56"
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="y6BziE3fp5q4" outputId="ce70af0d-e54a-44e1-911d-8b87505b980f"
ypred_LinReg = sc.inverse_transform(LinReg.predict(x_test).reshape(-1,1))   #do the inverse to return to the original value (because it was previously standardized)
y_pred_LinReg = pd.DataFrame(ypred_LinReg)
y_pred_LinReg.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="5aXKGdvmqTYf" outputId="7b1b32c2-13f8-4f26-c517-d45d53968764"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test), hist=False, label='Actual')
sns.distplot(y_pred_LinReg, hist=False, label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="_83JnfEC8950" outputId="605fe750-383f-4d14-bc1b-8066532196b4"
#Check the accuracy of testing and training of the Linear Regression model
print('Linear Regression')
LinReg_train = LinReg.score(x_train,y_train)*100
LinReg_test = LinReg.score(x_test, y_test)*100

#Assess the performance of the Linear Regression method by dividing the sample data by 10 folds randomly
LinReg_cv = KFold(n_splits=10, random_state=0, shuffle=True)
LinReg_score = cross_val_score(LinReg,x,y,cv=LinReg_cv)

print('Train : ',LinReg_train)
print('Test  : ',LinReg_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(LinReg_score)*100,2))

# + [markdown] id="PXYbS0eAA0L1"
# ### Hyperparameter Tuning

# + colab={"base_uri": "https://localhost:8080/"} id="qhQar6WEA3wJ" outputId="85769932-c370-4f1a-b539-1bb9efa2393c"
LinReg.get_params()

# + colab={"base_uri": "https://localhost:8080/"} id="W1Qz8QYkBHZS" outputId="5ca17c2c-818d-4f97-d4a9-bbe5035ebd5a"
param_grid = dict(
    copy_X=[True, False],
    fit_intercept=[True, False],
    n_jobs=np.arange(1,11), 
    positive=[True, False],
)

LinReg_tuning = LinearRegression()

LinReg_search = GridSearchCV(estimator=LinReg_tuning,
                           param_grid=param_grid,
                           scoring='r2')

LinReg_best_model = LinReg_search.fit(x_train, y_train)
print('Optimum parameters', LinReg_best_model.best_params_)
print('Best score is {}'.format(LinReg_best_model.best_score_))

# + [markdown] id="OCwuPkeMKRFZ"
# ### Fit New Model

# + colab={"base_uri": "https://localhost:8080/"} id="L7PT4NJPKM-r" outputId="bdd54678-a87b-4d86-e34e-2a1b6fd24786"
from sklearn.linear_model import LinearRegression
LinReg_model = LinearRegression(copy_X= True, fit_intercept= False, n_jobs= 1, positive= True)
LinReg_model.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="YLGHY3ryKM_A" outputId="b000aad2-b073-403d-a07b-78260f1eee05"
ypred_LinReg_model = sc.inverse_transform(LinReg_model.predict(x_test).reshape(-1,1))
ypred_LinReg_model = pd.DataFrame(ypred_LinReg_model)
ypred_LinReg_model.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="EX7vVbbtKM_C" outputId="cb51ab27-7988-486d-a7c5-c3ca51ef4de2"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(ypred_LinReg_model,hist=False,label='Predicted')
plt.legend()

# + [markdown] id="EynYU6tVywbx"
# ### Fit New Model

# + colab={"base_uri": "https://localhost:8080/"} id="7qbos8Qsywby" outputId="694d8cd6-10ad-4019-b202-031383b281dc"
from sklearn.ensemble import GradientBoostingRegressor
GBReg_model = GradientBoostingRegressor(
    subsample=0.1, 
    n_estimators=2000, 
    min_samples_split=9, 
    min_samples_leaf=1, 
    max_depth=10, 
    learning_rate=0.01, 
    random_state=0
)

# Fit model pada data training
GBReg_model.fit(x_train, y_train.ravel()) 

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="kEq3OuaEywby" outputId="f662445c-369e-4430-a2eb-e2d9bfe0f7cb"
ypred_GBReg_model = sc.inverse_transform(GBReg_model.predict(x_test).reshape(-1,1))
ypred_GBReg_model = pd.DataFrame(ypred_GBReg_model)
ypred_GBReg_model.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="fVb8Tb0xywby" outputId="a4ea8b26-7010-4f61-9df6-6292ebdfde48"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(ypred_GBReg_model,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="_af3sM_kywbz" outputId="ced6e339-4b9f-4776-97c7-fcb875d41e7b"
#Check the accuracy of testing and training of the Gradient Boosting Regression model
print('Gradient Boosting Regression')
GBReg_model_train = GBReg_model.score(x_train,y_train.ravel())*100
GBReg_model_test = GBReg_model.score(x_test, y_test.ravel())*100

#Assess the performance of the Gradient Boosting Regression method by dividing the sample data by 10 folds randomly
GBReg_model_cv = KFold(n_splits=10, random_state=0, shuffle=True)
GBReg_model_score = cross_val_score(GBReg_model,x,y,cv=GBReg_model_cv)

print('Train : ',GBReg_model_train)
print('Test  : ',GBReg_model_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(GBReg_model_score)*100,2))

# + [markdown] id="3DlEF8j9ywbz"
# The new regression model with the gradient boosting algorithm has decreased their accuracy in the train data but increased in the test.

# + [markdown] id="7XlLsb27Vagu"
# ## Support Vector Regression

# + [markdown] id="Tn7o7LIA2JFx"
# SVR gives the flexibility to define how much error is acceptable in model and will find an appropriate line (or hyperplane in higher dimensions) to fit the data.

# + colab={"base_uri": "https://localhost:8080/"} id="xA8-UyUr8gXc" outputId="7009ab7d-d508-4273-eb14-daeec924f41e"
from sklearn.svm import SVR
SVReg = SVR(gamma = 1)
SVReg.fit(x_train, y_train.ravel())

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="iz5RCBYsFDps" outputId="37d0b1e7-689d-4295-ea09-db8d00de74bb"
ypred_SVReg = sc.inverse_transform(SVReg.predict(x_test).reshape(-1,1))
y_pred_SVReg = pd.DataFrame(ypred_SVReg)
y_pred_SVReg.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="H1raXG-PFSOX" outputId="562ae61d-c21d-4886-976d-96b1c6d7ba67"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(y_pred_SVReg,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="7AXnPzV-FcZz" outputId="66163f6c-7db8-40ed-8ab1-19e5be315425"
#Check the accuracy of testing and training of the Support Vector Regression model
print('Support Vector Regression')
SVReg_train = SVReg.score(x_train,y_train.ravel())*100
SVReg_test = SVReg.score(x_test, y_test)*100

#Assess the performance of the Support Vector Regression method by dividing the sample data by 10 folds randomly
SVReg_cv = KFold(n_splits=10, random_state=0, shuffle=True)
SVReg_score = cross_val_score(SVReg,x,y,cv=SVReg_cv)

print('Train : ',SVReg_train)
print('Test  : ',SVReg_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(SVReg_score)*100,2))

# + [markdown] id="1WrxroLTBhsz"
# ### Hyperparameter Tuning

# + colab={"base_uri": "https://localhost:8080/"} id="qwm77FCyBmcd" outputId="9fa453de-5534-4750-8ae3-558c817258a0"
SVReg.get_params()

# + colab={"base_uri": "https://localhost:8080/"} id="u8O4VbyHB6Lq" outputId="181bc448-07d8-46ba-86f8-807c34a7d2e8"
param_grid = {'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'C': [0.1, 1, 10, 100, 1000],
              'epsilon': [0.001, 0.01, 0.1]}             

SVReg_tuning = SVR()

SVReg_search = GridSearchCV(SVReg_tuning, param_grid, cv = 10, scoring='r2')

SVReg_best_model = SVReg_search.fit(x_train, y_train.ravel())
print('Optimum parameters', SVReg_best_model.best_params_)
print('Best score is {}'.format(SVReg_best_model.best_score_))

# + [markdown] id="0wHKL0VLEl2e"
# ### Fit New Model

# + colab={"base_uri": "https://localhost:8080/"} id="3OvErB2eEpRB" outputId="148349bb-495e-4c3a-ee71-db837ea6f2cc"
SVReg_model = SVR(C= 1, epsilon= 0.1, gamma= 0.1)
SVReg_model.fit(x_train, y_train.ravel())

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="YO6Rrwr1FMcX" outputId="ff66f30b-765c-4b9e-d573-1c26bf9aceb5"
ypred_SVReg_model = sc.inverse_transform(SVReg_model.predict(x_test).reshape(-1,1))
ypred_SVReg_model = pd.DataFrame(ypred_SVReg_model)
ypred_SVReg_model.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="szyp4dMmFMcY" outputId="6eaa7c3c-89eb-4be7-c879-dcbabc4206e7"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(ypred_SVReg_model,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="H6h4_JUpFMcZ" outputId="5aef4bec-b557-44ab-fdf4-dad1179c6a92"
#Check the accuracy of testing and training of the Support Vector Regression model
print('Support Vector Regression')
SVReg_model_train = SVReg_model.score(x_train,y_train.ravel())*100
SVReg_model_test = SVReg_model.score(x_test, y_test)*100

#Assess the performance of the Support Vector Regression method by dividing the sample data by 10 folds randomly
SVReg_model_cv = KFold(n_splits=10, random_state=0, shuffle=True)
SVReg_model_score = cross_val_score(SVReg,x,y,cv=SVReg_model_cv)

print('Train : ',SVReg_model_train)
print('Test  : ',SVReg_model_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(SVReg_model_score)*100,2))

# + [markdown] id="4964y65YGoqi"
# The support vector regression model has decreased their accuracy in the train data but increased in the test, which means overfitting has minimized.

# + [markdown] id="fccZ2k6tVaU2"
# ## Decision Tree Regression

# + [markdown] id="0HA2JBc91GSo"
# Decision Tree Regressor is an algorithm for building a regression model that observes the features of an object and trains the model in a tree structure.

# + colab={"base_uri": "https://localhost:8080/"} id="0XgSoqqk8joY" outputId="034dc434-35a7-4238-b2d3-effdf1feb72d"
from sklearn.tree import DecisionTreeRegressor
DTreeReg = DecisionTreeRegressor(random_state = 0)
DTreeReg.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="AEL2pnE4Gyah" outputId="27a7a86a-86eb-457c-a616-567fede390bb"
ypred_DTreeReg = sc.inverse_transform(DTreeReg.predict(x_test).reshape(-1,1))
y_pred_DTreeReg = pd.DataFrame(ypred_DTreeReg)
y_pred_DTreeReg.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="v47ECFnAGya-" outputId="63d5e9d8-b882-4ab6-d963-9c486014056b"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(y_pred_DTreeReg,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="itxcXyWHGya-" outputId="f98d100a-fbb0-4577-9460-513dd32527ed"
#Check the accuracy of testing and training of the Decision Tree Regression model
print('Decision Tree Regression')
DTreeReg_train = DTreeReg.score(x_train,y_train)*100
DTreeReg_test = DTreeReg.score(x_test, y_test)*100

#Assess the performance of the Decision Tree Regression method by dividing the sample data by 10 folds randomly
DTreeReg_cv = KFold(n_splits=10, random_state=0, shuffle=True)
DTreeReg_score = cross_val_score(DTreeReg,x,y,cv=DTreeReg_cv)

print('Train : ',DTreeReg_train)
print('Test  : ',DTreeReg_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(DTreeReg_score)*100,2))

# + [markdown] id="NeRzEaV5HTbw"
# ### Hyperparameter Tuning

# + colab={"base_uri": "https://localhost:8080/"} id="1GChJg7wHW2m" outputId="0f02d994-50c5-4a4f-8b2e-8f69874d0d3e" active=""
# DTreeReg.get_params()

# + colab={"base_uri": "https://localhost:8080/"} id="Ck6PjFOiHsBM" outputId="235d95f1-6b34-43b9-98ae-1bb52c030357"
#param_grid = {'splitter' : ['best', 'random'],
#              'max_depth': np.arange(1,11),
#              'min_samples_leaf': np.arange(1, 11),
#              'min_samples_split': np.arange(2, 11),
#              'max_features':['auto','log2','sqrt',None]}             

#DTreeReg_tuning = DecisionTreeRegressor()

#DTreeReg_search = RandomizedSearchCV(DTreeReg_tuning, param_grid, cv = 10, scoring='r2')

#DTreeReg_best_model = DTreeReg_search.fit(x_train, y_train)
#print('Optimum parameters', DTreeReg_best_model.best_params_)
#print('Best score is {}'.format(DTreeReg_best_model.best_score_))

# + [markdown] id="rSarA85yJ0dO"
# ### Fit New Model

# + colab={"base_uri": "https://localhost:8080/"} id="hVqSccILJ4HZ" outputId="772ae98c-62cd-4855-878e-edcc790c1c93"
DTreeReg_model = DecisionTreeRegressor(
    splitter='best', 
    min_samples_split=8, 
    min_samples_leaf=5, 
    max_features=None,  # Menggunakan semua fitur
    max_depth=2, 
    random_state=0
)

# Melatih model pada data training
DTreeReg_model.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="GwRcSe3OKd22" outputId="2e0fe85d-35cc-4972-cd45-af32fadfeb2d"
ypred_DTreeReg_model = sc.inverse_transform(DTreeReg_model.predict(x_test).reshape(-1,1))
ypred_DTreeReg_model = pd.DataFrame(ypred_DTreeReg_model)
ypred_DTreeReg_model.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="5upYuD9LKd29" outputId="f896f82c-09dd-493f-e7b3-f8622119434b"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(ypred_DTreeReg_model,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="swwuODV9Kd29" outputId="f41df597-de84-4469-ade5-6aba441831de"
#Check the accuracy of testing and training of the Decision Tree Regression model
print('Decision Tree Regression')
DTreeReg_model_train = DTreeReg_model.score(x_train,y_train)*100
DTreeReg_model_test = DTreeReg_model.score(x_test, y_test)*100

#Assess the performance of the Decision Tree Regression method by dividing the sample data by 10 folds randomly
DTreeReg_model_cv = KFold(n_splits=10, random_state=0, shuffle=True)
DTreeReg_model_score = cross_val_score(DTreeReg_model,x,y,cv=DTreeReg_model_cv)

print('Train : ',DTreeReg_model_train)
print('Test  : ',DTreeReg_model_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(DTreeReg_model_score)*100,2))

# + [markdown] id="D2uLNBa5LnwQ"
# The Decision Tree regression model that has used the results of hyperparameter tuning shows that the performance is more balanced between the train and the test data.

# + [markdown] id="qd_zD2ptVa2r"
# ## K-Neighbors Regression

# + [markdown] id="7DvzIu3S07di"
# KNN Regressor is an algorithm for building a regression model that uses the average or median value of k neighbors to predict the target element.

# + colab={"base_uri": "https://localhost:8080/"} id="ENR_vXOO8m_K" outputId="c0e5f6f3-15e8-4a95-aa61-84948afab55d"
from sklearn.neighbors import KNeighborsRegressor
KNNReg = KNeighborsRegressor(n_neighbors=1)
KNNReg.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="uofGTPJyH96N" outputId="9420dc51-abde-4ece-d0d6-03e056069158"
ypred_KNNReg = sc.inverse_transform(KNNReg.predict(x_test).reshape(-1,1))
y_pred_KNNReg = pd.DataFrame(ypred_KNNReg)
y_pred_KNNReg.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="K63uG0foH96o" outputId="44543bea-7246-46bd-b47c-a070e37cd646"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(y_pred_KNNReg,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="YRPeIF3uH96p" outputId="2c0e78b2-28b5-4bb7-d8dc-dcf492120f03"
#Check the accuracy of testing and training of the K-Nearest Neighbors Regression model
print('K-Nearest Neighbors Regression')
KNNReg_train = KNNReg.score(x_train,y_train)*100
KNNReg_test = KNNReg.score(x_test, y_test)*100

#Assess the performance of the K-Nearest Neighbor Regression method by dividing the sample data by 10 folds randomly
KNNReg_cv = KFold(n_splits=10, random_state=0, shuffle=True)
KNNReg_score = cross_val_score(KNNReg,x,y,cv=KNNReg_cv)

print('Train : ',KNNReg_train)
print('Test  : ',KNNReg_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(KNNReg_score)*100,2))

# + [markdown] id="2cTa4Y1oMAgt"
# ### Hyperparameter Tuning

# + colab={"base_uri": "https://localhost:8080/"} id="hQSdxeNeMFPD" outputId="0cc568d9-2cbb-4079-d0d2-f43743e908e5"
KNNReg.get_params()

# + colab={"base_uri": "https://localhost:8080/"} id="Ca8kCsqq4UrN" outputId="25316168-60f7-4f80-ff4b-005dfda021a8"
param_grid = {'n_neighbors': np.arange(1, 10)}

KNNReg_tuning = KNeighborsRegressor()

KNNReg_search = GridSearchCV(KNNReg_tuning, param_grid, cv=10, scoring='r2')

KNNReg_best_model = KNNReg_search.fit(x_train, y_train)
print('Optimum parameters', KNNReg_best_model.best_params_)
print('Best score is {}'.format(KNNReg_best_model.best_score_))

# + [markdown] id="SuX-O4BiN1Yr"
# ### Fit New Model

# + colab={"base_uri": "https://localhost:8080/"} id="3PIbf5PEN6CJ" outputId="a2673e6b-318a-4700-bc7e-94ba519a36c9"
KNNReg_model = KNeighborsRegressor(n_neighbors= 3)
KNNReg_model.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="9UlcHajrOROE" outputId="dd2cfc13-8be7-4cdd-a833-abc132020aa1"
ypred_KNNReg_model = sc.inverse_transform(KNNReg_model.predict(x_test).reshape(-1,1))
ypred_KNNReg_model = pd.DataFrame(ypred_KNNReg_model)
ypred_KNNReg_model.tail()

# + colab={"base_uri": "https://localhost:8080/", "height": 396} id="wkpYUvL7OROG" outputId="48ed78cd-b2de-4796-df63-ca3e2354ed87"
#visualize the prediction
sns.distplot(sc.inverse_transform(y_test),hist=False,label='Actual')
sns.distplot(ypred_KNNReg_model,hist=False,label='Predicted')
plt.legend()

# + colab={"base_uri": "https://localhost:8080/"} id="uvU4wewfOROG" outputId="09f459d3-cc95-47d3-c3c4-12b6239e3a09"
#Check the accuracy of testing and training of the K-Nearest Neighbors Regression model
print('K-Nearest Neighbors Regression')
KNNReg_model_train = KNNReg_model.score(x_train,y_train)*100
KNNReg_model_test = KNNReg_model.score(x_test, y_test)*100

#Assess the performance of the K-Nearest Neighbor Regression method by dividing the sample data by 10 folds randomly
KNNReg_model_cv = KFold(n_splits=10, random_state=0, shuffle=True)
KNNReg_model_score = cross_val_score(KNNReg_model,x,y,cv=KNNReg_model_cv)

print('Train : ',KNNReg_model_train)
print('Test  : ',KNNReg_model_test, '\n')
print('The Average Cross Validation Score is',np.round(np.mean(KNNReg_model_score)*100,2))

# + [markdown] id="nIEDQ87FO8zK"
# After hyperparameter tuning, the new model no longer tends to overfit.

# + [markdown] id="HotjuFaIJjhV"
# # 5. Model Evaluation

# + [markdown] id="UAnaNUg6Kn92"
# The evaluation below is carried out using the mean absolute error, mean squared error, and r2-score. The smaller the mean absolute error and mean squared error, but the greater the R2-score, the better the algorithm.
#
# 1. R2-score (coefficient of determination) regression score function will represents the proportion of the variance for items (crops) in the regression model. R2-score shows how well terms (data points) fit a curve or line.
#
# 2. The Mean absolute error (MAE) represents the average of the absolute difference between the actual and predicted values in the dataset. It measures the average of the residuals in the dataset.
#
# 3. Mean Squared Error (MSE) represents the average of the squared difference between the original and predicted values in the data set. It measures the variance of the residuals.

# + id="nQ-V3LffJqM5"
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# + colab={"base_uri": "https://localhost:8080/", "height": 238} id="6bt5gnqBKu-P" outputId="4c75e5ed-a672-4910-8719-f91d6fbace71"
results= pd.DataFrame(columns=['R2-score','Mean Absolute Error','Mean Squared Error'])
results.loc['Linear Regression']=[r2_score(sc.inverse_transform(y_test),ypred_LinReg_model)*100,
                                  mean_absolute_error(sc.inverse_transform(y_test),ypred_LinReg_model),
                                  mean_squared_error(sc.inverse_transform(y_test),ypred_LinReg_model)]

results.loc['Gradient Boosting']=[r2_score(sc.inverse_transform(y_test),ypred_GBReg_model)*100,
                                  mean_absolute_error(sc.inverse_transform(y_test),ypred_GBReg_model),
                                  mean_squared_error(sc.inverse_transform(y_test),ypred_GBReg_model)]
results.loc['SVR']=[r2_score(sc.inverse_transform(y_test),ypred_SVReg_model)*100,
                    mean_absolute_error(sc.inverse_transform(y_test),ypred_SVReg_model),
                    mean_squared_error(sc.inverse_transform(y_test),ypred_SVReg_model)]
results.loc['Decision Tree']=[r2_score(sc.inverse_transform(y_test),ypred_DTreeReg_model)*100,
                              mean_absolute_error(sc.inverse_transform(y_test),ypred_DTreeReg_model),
                              mean_squared_error(sc.inverse_transform(y_test),ypred_DTreeReg_model)]
results.loc['K-Neighbors']=[r2_score(sc.inverse_transform(y_test),ypred_KNNReg_model)*100,
                            mean_absolute_error(sc.inverse_transform(y_test),ypred_KNNReg_model),
                            mean_squared_error(sc.inverse_transform(y_test),ypred_KNNReg_model)]

#Sorts models based on R2-score
results.sort_values('R2-score',ascending=False).style.background_gradient(cmap='Purples', subset=['R2-score'])

# + [markdown] id="6AuvrEQItUjT"
# From results viewed above, model with Linear Regression algorithm has the highest R2-score - 86.9%. So that, this model can be the best choice for use in predicting agricultural production in Sumatra, in accordance with the objectives described earlier.
# -

import streamlit as st
import joblib
import numpy as np

# !pip install streamlit

model = joblib.load('Data_Tanaman_Padi_Sumatera_version_1.pkl')


