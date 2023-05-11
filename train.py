import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

from tensorflow.keras import layers
from tensorflow import keras
warnings.filterwarnings('ignore')

from utils.data_load import interpolate_outliers, normalizado, datosstack, split_sequences
from utils.transformer import build_model

np.random.seed(0)


wind_turbine = 'WT2346'
df = pd.read_csv(f'./Data/SCADA_data_{wind_turbine}.csv')
df = df[['date_time', 'TempEjeLento_1', 'TempAmbMean', 'TempRodamMultipMean', 'TempCojLAMean', 'TempCojLOAMean', 'TempGenMean', 'PotMean', 'VelRotorMean']]

df = interpolate_outliers(df)

datos_listos = df.copy()

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


# Model creation and training
input_shape = xtrain.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[100],
    #mlp_dropout=0.2,
    dropout=0.3,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
)

callbacks = [keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)]

history = model.fit(
    xtrain,
    ytrain,
    #validation_split=0.2,
    validation_data=(xvalidation, yvalidation),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
)

loss = history.history['loss']
val_loss = history.history['val_loss']


sns.set_theme(palette="ch:s=.25,rot=-.25")
fig,ax = plt.subplots(figsize=(8,8))
sns.lineplot(data=loss, ax = ax, color="b", label='Training Loss')
sns.lineplot(data=val_loss, ax = ax, color="r", label='Validation Loss')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.show()
plt.savefig(f"./plots/model_{wind_turbine}_v3.png")


model.save(f'./Models/model_{wind_turbine}_v4.h5')

