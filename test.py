import seaborn as sns
import numpy as np
from numpy import array
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from sklearn import metrics 
from tensorflow import keras

from utils.data_load import interpolate_outliers, normalizado, datosstack, split_sequences

np.random.seed(0)
warnings.filterwarnings('ignore')

filename = 'WT2340'
df = pd.read_csv('./Data/SCADA_data_{}.csv'.format(filename))
df = df[['date_time', 'TempEjeLento_1', 'TempAmbMean', 'TempRodamMultipMean', 'TempCojLAMean', 'TempCojLOAMean', 'TempGenMean', 'PotMean', 'VelRotorMean']]


df = interpolate_outliers(df)

datos_listos=df.copy()

datos_t=pd.DataFrame({ 'TempEjeLento_1':df['TempEjeLento_1'],'date_time': df['date_time'] ,})
datos_t.drop([0], inplace=True)
datos_t.reset_index(drop=True, inplace=True)

datos_tmenos1=pd.DataFrame({'TempEjeLento_1-1': df['TempEjeLento_1'],'TempAmbMean-1': df['TempAmbMean'], 'TempRodamMultipMean-1': df['TempRodamMultipMean'],
                            'TempCojLOAMean-1': df['TempCojLOAMean'], 'TempGenMean-1': df['TempGenMean'], 'PotMean-1': df['PotMean'], 'VelRotorMean-1': df['VelRotorMean']})
datos_tmenos1.drop([len(df['VelRotorMean'])-1], inplace=True)
datos_tmenos1.reset_index(drop=True, inplace=True)


datos_listos=pd.DataFrame({'date_time': datos_t['date_time'], 'TempEjeLento_1-1': datos_tmenos1['TempEjeLento_1-1'],'TempAmbMean-1': datos_tmenos1['TempAmbMean-1'],
                           'TempRodamMultipMean-1': datos_tmenos1['TempRodamMultipMean-1'],'TempCojLOAMean-1': datos_tmenos1['TempCojLOAMean-1'],
                           'TempGenMean-1': datos_tmenos1['TempGenMean-1'], 'PotMean-1': datos_tmenos1['PotMean-1'], 'VelRotorMean-1': datos_tmenos1['VelRotorMean-1'],
                           'TempEjeLento_1': datos_t['TempEjeLento_1']})


from datetime import datetime,timedelta
datos_listos['date_time']=pd.to_datetime(datos_listos['date_time'])


data_tv=datos_listos.copy()
mask = ((data_tv['date_time'] >= '2017-02-06 00:00:00') & (data_tv['date_time'] < '2018-01-01 00:00:00') ) 
train_tv=data_tv.loc[mask]
train_tv.reset_index(drop=True, inplace=True)


longitud=len(train_tv)
longitud=(longitud)/144
longitud=longitud*90/100
longitud=round(longitud)
longitud=longitud*144

training=train_tv.loc[0:round(longitud)-1]
training.reset_index(drop=True, inplace=True)

df=training

a1=df['TempEjeLento_1-1'].values.max()
b1=df['TempEjeLento_1-1'].values.min()
a2=df['TempAmbMean-1'].values.max()
b2=df['TempAmbMean-1'].values.min()
a6=df['TempRodamMultipMean-1'].values.max()
b6=df['TempRodamMultipMean-1'].values.min()
a10=df['TempCojLOAMean-1'].values.max()
b10=df['TempCojLOAMean-1'].values.min()
a12=df['TempGenMean-1'].values.max()
b12=df['TempGenMean-1'].values.min()
a14=df['PotMean-1'].values.max()
b14=df['PotMean-1'].values.min()
a16=df['VelRotorMean-1'].values.max()
b16=df['VelRotorMean-1'].values.min()
a17=df['TempEjeLento_1'].values.max()
b17=df['TempEjeLento_1'].values.min()


training=normalizado(training,a1,b1,a2,b2,a6,b6,a10,b10,a12,b12,a14,b14,a16,b16,a17,b17)


validation=train_tv.loc[round(longitud):len(train_tv)]
validation.reset_index(drop=True, inplace=True)
validation=normalizado(validation,a1,b1,a2,b2,a6,b6,a10,b10,a12,b12,a14,b14,a16,b16,a17,b17)


test=datos_listos.copy()
mask = ((test['date_time'] >= '2018-01-01 00:00:00') & (test['date_time'] < '2018-12-01 00:00:00')) 
test=test.loc[mask]
test.reset_index(drop=True, inplace=True)
test=normalizado(test,a1,b1,a2,b2,a6,b6,a10,b10,a12,b12,a14,b14,a16,b16,a17,b17)


dataset_train=datosstack(training)
dataset_validation=datosstack(validation)
dataset_test=datosstack(test)


n_steps = 144
xtrain, ytrain = split_sequences(dataset_train, n_steps)
xvalidation, yvalidation = split_sequences(dataset_validation, n_steps)
xtest, ytest = split_sequences(dataset_test, n_steps)

# Model evaluation
model = keras.models.load_model('./Models/model_{}_v2.h5'.format(filename))


pred = model.predict(xtest)
score = np.sqrt(metrics.mean_squared_error(pred,ytest))
print("Score RMSE: {}".format(score))


def diferencia(predict,test):
    ypred = []
    for data in predict:
      ypred.append(data[0])
    df_test = pd.DataFrame({'Real': list(test), 'Prediction': ypred})
    df_test['Difference'] = (df_test['Real'] - df_test['Prediction']).abs()
    return  df_test


df_test=diferencia(pred,ytest)


# Data merging
xfulldata = np.concatenate((xtrain,xvalidation,xtest))
yfulldata = np.concatenate((ytrain,yvalidation,ytest))

yfullpred = model.predict(xfulldata)

df_full=diferencia(yfullpred,yfulldata)


sns.set_theme(style = "darkgrid")
fig, axs = plt.subplots(figsize =(20, 10))
sns.lineplot(data =df_full, x=df_full.index, y="Real", ax=axs, color="b")
sns.lineplot(data =df_full, x=df_full.index, y="Prediction", ax=axs, color="r")
plt.show()

sns.set_theme(style = "darkgrid")
fig, axs = plt.subplots(figsize =(20, 5))
sns.lineplot(data =df_full['Difference'], ax=axs, color='r')
plt.show()

def moving_average(df, value):
    #list_ma = df.rolling(value,min_periods=1).mean()
    list_ma = df.rolling(value).mean()
    return list_ma

# split a multivariate sequence into samples
def split_sequencesfecha(sequences, n_steps):
    X = list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x= sequences[end_ix-1]
        X.append(seq_x)
    return array(X)


fechatrain=split_sequencesfecha(training['date_time'], 144)
fechaval=split_sequencesfecha(validation['date_time'], 144)
fechatest=split_sequencesfecha(test['date_time'], 144)
fechafull = np.concatenate((fechatrain,fechaval,fechatest))


tiempo = pd.DataFrame({'date_time': fechafull})


x2metr = np.concatenate((xtrain,xvalidation))


def ma(diferencia,numdatos):
    aux = pd.DataFrame(diferencia)
    #aux['Anomaly'] = pd.DataFrame(diferencia)
    auxma=moving_average(aux, numdatos)
    diferencia2=auxma[0:len(x2metr)-1]
    meandiff=np.mean(diferencia2)
    stdiff=np.std(diferencia2)
    tiempo["diferencia"]=auxma
    tiempo["target"]=0
    thr1 = meandiff + 6*stdiff
    thr2 = meandiff + 5*stdiff
    thr3 = meandiff + 3*stdiff
    thr4 = meandiff + 2*stdiff
    
    return auxma,meandiff,stdiff,thr1,thr2,thr3,thr4


def threshold():
    datatest1["threshold1"]=float(threshold1)
    datatest1["threshold2"]=float(threshold2)
    datatest1["threshold3"]=float(threshold3)
    datatest1["threshold4"]=float(threshold4)
    return 


def monthAgo(date):
    date = pd.to_datetime(date)
    month = date-timedelta(days=30)
    return month

def tagClass(dataframe,numClasses,startFailure):
    c=0.1
    dataframe=dataframe.copy()
    dataframe['date_time']=pd.to_datetime(dataframe['date_time'])
    for i in range(numClasses):
        if i==0:
            startDate=startFailure
            finishDate=monthAgo(startFailure)
            
            dataframe.loc[(dataframe['date_time']>=finishDate)&(dataframe['date_time']<startDate),'target']=0.1
        else:
            startDate=finishDate
            finishDate=monthAgo(finishDate)
            
            dataframe.loc[(dataframe['date_time']>=finishDate)&(dataframe['date_time']<startDate),'target']=c+0.1
        c+=0.1
    
    return dataframe


auxma1,meandiff1,stdiff1,threshold1,threshold2,threshold3,threshold4=ma(df_full['Difference'],144)


datatest1=tagClass(tiempo,6,pd.to_datetime("21/05/2018").replace(minute=0, hour=0, second=0))
threshold()

datatest1["meandiff"]=float(meandiff1)
datatest1["stdiff"]=float(stdiff1)


datatest1["meandiff"].shape

# %%
auxma1.shape

# %%
plt.figure(figsize=(20, 7))
plt.plot(auxma1, label='diferencia')
plt.plot(datatest1["meandiff"], label='meandiff1')
plt.plot(datatest1["stdiff"], label='stdiff')
plt.legend(frameon=False)

# %%
plt.savefig("Test2_mean.png")

# %%
def datatime_categorias():

    datatest1['categoria']=0
    #training
    fecha1=pd.to_datetime("2017/02/06").replace(minute=0, hour=0, second=0)
    #validation
    fecha2=pd.to_datetime(validation["date_time"][0]).replace(minute=0, hour=0, second=0)
    # fin training - validation
    fecha3=pd.to_datetime("2017/12/31").replace(minute=0, hour=0, second=0)
    ########
    fecha4=pd.to_datetime("2018/12/01").replace(minute=0, hour=0, second=0)

    datatest1.loc[(datatest1['date_time']>=fecha1)&(datatest1['date_time']<fecha2),'categoria']=200
    datatest1.loc[(datatest1['date_time']>=fecha2)&(datatest1['date_time']<=fecha3),'categoria']=300
    datatest1.loc[(datatest1['date_time']>fecha3),'categoria']=400


    return 

# %%
import matplotlib.dates as mdates

# %%
threshold()
datatime_categorias()
plt.figure(figsize=(20, 7))
plt.plot(auxma1, label='diferencia')
plt.plot(datatest1["threshold1"], label='$\mathrm{\mathbb{\kappa}}$=6')
plt.plot(datatest1["threshold3"], label='$\mathrm{\mathbb{\kappa}}$=3')

plt.legend(loc = 'upper left')

# %%
threshold()
datatime_categorias()

WT="WT1"
df1=datatest1[['date_time','diferencia','target','categoria']]
datosinindicador=datatest1
df2=datatest1[['date_time','diferencia','target','categoria']]

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(40, 25))

plt.plot_date(datatest1[datatest1["categoria"]==200]['date_time'], datatest1[datatest1["categoria"]==200]['diferencia'], color="#198bf2", marker=".", markersize=1, linestyle='solid',linewidth=5,label='FPI (train) WT2')
plt.plot_date(datatest1[datatest1["categoria"]==300]['date_time'], datatest1[datatest1["categoria"]==300]['diferencia'], color="green", marker=".", markersize=1, linestyle='solid',linewidth=5,label='FPI (validation) WT2')
plt.plot_date(datatest1[datatest1["categoria"]==400]['date_time'], datatest1[datatest1["categoria"]==400]['diferencia'], color="#6a329f", marker=".", markersize=1, linestyle='solid',linewidth=5,label='FPI (test) WT2')
plt.plot_date(datatest1["date_time"], datatest1["threshold1"], color="red", marker=".", markersize=0.1, linestyle='dashed',linewidth=10,label='$\mathrm{\mathbb{\kappa}}$=6')
plt.plot_date(datatest1["date_time"], datatest1["threshold3"], color="#e1ad10", marker=".", markersize=0.1, linestyle='dashed',linewidth=10,label='$\mathrm{\mathbb{\kappa}}$=3')

plt.axvline(pd.to_datetime('2018-05-21 23:50:00'), 0, 1,color="black",linewidth=2)

tick_spacing = 20

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.DayLocator(interval=150))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))

start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(0, end+0.1, 0.2))

plt.xticks(fontsize=40)
plt.yticks(fontsize=30)

plt.legend(fontsize=30,loc = 'upper left')
plt.gcf().autofmt_xdate()
plt.tick_params(which='both', width=5, length=5)

# %%
auxma3,meandiff3,stdiff3,threshold1,threshold2,threshold3,threshold4=ma(df_full['Difference'],1008)

from datetime import datetime,timedelta
tiempo['date_time']=pd.to_datetime(tiempo['date_time'])
datatest1=tagClass(tiempo,6,pd.to_datetime("21/05/2018").replace(minute=0, hour=0, second=0))
threshold()

datatest1["meandiff"]=float(meandiff3)
datatest1["stdiff"]=float(stdiff3)

# %%
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 7))
plt.plot(auxma3, label='diferencia')
plt.plot(datatest1["meandiff"], label='meandiff1')
plt.plot(datatest1["stdiff"], label='stdiff')
plt.legend(frameon=False)

# %%
threshold()
datatime_categorias()

WT="WT1"
df1=datatest1[['date_time','diferencia','target','categoria']]
datosinindicador=datatest1
df2=datatest1[['date_time','diferencia','target','categoria']]

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
plt.figure(figsize=(40, 25))
plt.style.use('classic')

plt.plot_date(datatest1[datatest1["categoria"]==200]['date_time'], datatest1[datatest1["categoria"]==200]['diferencia'], color="#198bf2", marker=".", markersize=1, linestyle='solid',linewidth=5,label='FPI (train) WT2')
plt.plot_date(datatest1[datatest1["categoria"]==300]['date_time'], datatest1[datatest1["categoria"]==300]['diferencia'], color="green", marker=".", markersize=1, linestyle='solid',linewidth=5,label='FPI (validation) WT2')
plt.plot_date(datatest1[datatest1["categoria"]==400]['date_time'], datatest1[datatest1["categoria"]==400]['diferencia'], color="#6a329f", marker=".", markersize=1, linestyle='solid',linewidth=5,label='FPI (test) WT2')
plt.plot_date(datatest1["date_time"], datatest1["threshold1"], color="red", marker=".", markersize=0.1, linestyle='dashed',linewidth=10,label='$\mathrm{\mathbb{\kappa}}$=6')
plt.plot_date(datatest1["date_time"], datatest1["threshold3"], color="#e1ad10", marker=".", markersize=0.1, linestyle='dashed',linewidth=10,label='$\mathrm{\mathbb{\kappa}}$=3')

plt.axvline(pd.to_datetime('2018-05-21 23:50:00'), 0, 1,color="black",linewidth=2)

tick_spacing = 20

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.DayLocator(interval=150))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))

start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(0, end+0.02, 0.05))

plt.xticks(fontsize=50)
plt.yticks(fontsize=50)

plt.legend(fontsize=30,loc = 'lower left')
plt.gcf().autofmt_xdate()
plt.tick_params(which='both', width=5, length=5)
#plt.savefig('../Testing plots/test_WT2346_v3.png')

# %%
threshold()
datatime_categorias()

WT="WT1"
df1=datatest1[['date_time','diferencia','target','categoria']]
datosinindicador=datatest1
df2=datatest1[['date_time','diferencia','target','categoria']]

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
plt.figure(figsize=(40, 25))
plt.style.use('classic')

plt.plot_date(datatest1[datatest1["categoria"]==200]['date_time'], datatest1[datatest1["categoria"]==200]['diferencia'], color="#198bf2", marker=".", markersize=1, linestyle='solid',linewidth=5)
plt.plot_date(datatest1[datatest1["categoria"]==300]['date_time'], datatest1[datatest1["categoria"]==300]['diferencia'], color="green", marker=".", markersize=1, linestyle='solid',linewidth=5)
plt.plot_date(datatest1[datatest1["categoria"]==400]['date_time'], datatest1[datatest1["categoria"]==400]['diferencia'], color="#6a329f", marker=".", markersize=1, linestyle='solid',linewidth=5)
plt.plot_date(datatest1["date_time"], datatest1["threshold1"], color="red", marker=".", markersize=0.1, linestyle='dashed',linewidth=10)
plt.plot_date(datatest1["date_time"], datatest1["threshold3"], color="#e1ad10", marker=".", markersize=0.1, linestyle='dashed',linewidth=10)

plt.axvline(pd.to_datetime('2018-05-21 23:50:00'), 0, 1,color="black",linewidth=2)

tick_spacing = 20

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.DayLocator(interval=150))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))

start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(0, end+0.02, 0.05))

plt.xticks(fontsize=40, rotation=45)
plt.yticks(fontsize=40)

#plt.legend(fontsize=30,loc = 'lower left')
plt.gcf().autofmt_xdate()
plt.tick_params(which='both', width=5, length=5)
plt.savefig('../plots/test_{}.png'.format(filename))

# %%



