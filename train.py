import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

from tensorflow.keras import layers
from tensorflow import keras
warnings.filterwarnings('ignore')

np.random.seed(0)


wind_turbine = 'WT2346'
df = pd.read_csv(f'./Data/SCADA_data_{wind_turbine}.csv')
df = df[['date_time', 'TempEjeLento_1', 'TempAmbMean', 'TempRodamMultipMean', 'TempCojLAMean', 'TempCojLOAMean', 'TempGenMean', 'PotMean', 'VelRotorMean']]

# Removing outliers and anomalous data
df['PotMean'][df['PotMean']< 0] = None
df['PotMean'][df['PotMean']> 2000] = None

df['TempAmbMean'][df['TempAmbMean']< -5] = None
df['TempAmbMean'][df['TempAmbMean']> 40] = None

df['TempCojLAMean'][df['TempCojLAMean']< 0] = None
df['TempCojLAMean'][df['TempCojLAMean']> 120] = None

df['TempCojLOAMean'][df['TempCojLOAMean']< 0] = None
df['TempCojLOAMean'][df['TempCojLOAMean']> 120] = None

df['TempEjeLento_1'][df['TempEjeLento_1']< 0] = None
df['TempEjeLento_1'][df['TempEjeLento_1']> 120] = None

df['TempGenMean'][df['TempGenMean']< 0] = None
df['TempGenMean'][df['TempGenMean']> 175] = None

df['TempRodamMultipMean'][df['TempRodamMultipMean']< 0] = None
df['TempRodamMultipMean'][df['TempRodamMultipMean']> 120] = None

df['VelRotorMean'][df['VelRotorMean']< 0] = None
df['VelRotorMean'][df['VelRotorMean']> 50] = None


# Interpolation to fill nan values
df['TempAmbMean']=df['TempAmbMean'].interpolate(method='pchip', order=3, limit_area='inside')
df['TempAmbMean']=df['TempAmbMean'].fillna(method='backfill')
df['TempAmbMean']=df['TempAmbMean'].fillna(method='ffill')

df['TempEjeLento_1']=df['TempEjeLento_1'].interpolate(method='pchip', order=3)
df['TempEjeLento_1']=df['TempEjeLento_1'].fillna(method='backfill')

df['TempRodamMultipMean']=df['TempRodamMultipMean'].interpolate(method='pchip', order=3)
df['TempRodamMultipMean']=df['TempRodamMultipMean'].fillna(method='backfill')

df['TempCojLAMean']=df['TempCojLAMean'].interpolate(method='pchip', order=3)
df['TempCojLAMean']=df['TempCojLAMean'].fillna(method='backfill')

df['TempCojLOAMean']=df['TempCojLOAMean'].interpolate(method='pchip', order=3)
df['TempCojLOAMean']=df['TempCojLOAMean'].fillna(method='backfill')

df['TempGenMean']=df['TempGenMean'].interpolate(method='pchip', order=3)
df['TempGenMean']=df['TempGenMean'].fillna(method='backfill')

df['PotMean']=df['PotMean'].interpolate(method='pchip', order=3)
df['PotMean']=df['PotMean'].fillna(method='backfill')

df['VelRotorMean']=df['VelRotorMean'].interpolate(method='pchip', order=3)
df['VelRotorMean']=df['VelRotorMean'].fillna(method='backfill')

df['TempEjeLento_1']=df['TempEjeLento_1'].fillna(method='ffill')
df['TempRodamMultipMean']=df['TempRodamMultipMean'].fillna(method='ffill')
df['TempCojLAMean']=df['TempCojLAMean'].fillna(method='ffill')
df['TempCojLOAMean']=df['TempCojLOAMean'].fillna(method='ffill')
df['TempGenMean']=df['TempGenMean'].fillna(method='ffill')
df['PotMean']=df['PotMean'].fillna(method='ffill')
df['VelRotorMean']=df['VelRotorMean'].fillna(method='ffill')

df.reset_index(drop=True, inplace=True)

datos_listos = df.copy()

datos_t=pd.DataFrame({ 'TempEjeLento_1':df['TempEjeLento_1'],'date_time': df['date_time'] ,})
datos_t.drop([0], inplace=True)
datos_t.reset_index(drop=True, inplace=True)
datos_t

datos_tmenos1=pd.DataFrame({'TempEjeLento_1-1': df['TempEjeLento_1'],'TempAmbMean-1': df['TempAmbMean'], 'TempRodamMultipMean-1': df['TempRodamMultipMean'],'TempCojLOAMean-1': df['TempCojLOAMean'], 'TempGenMean-1': df['TempGenMean'], 'PotMean-1': df['PotMean'], 'VelRotorMean-1': df['VelRotorMean']})
datos_tmenos1.drop([len(df['VelRotorMean'])-1], inplace=True)
datos_tmenos1.reset_index(drop=True, inplace=True)

datos_listos=pd.DataFrame({'date_time': datos_t['date_time'], 'TempEjeLento_1-1': datos_tmenos1['TempEjeLento_1-1'],'TempAmbMean-1': datos_tmenos1['TempAmbMean-1'],'TempRodamMultipMean-1': datos_tmenos1['TempRodamMultipMean-1'],'TempCojLOAMean-1': datos_tmenos1['TempCojLOAMean-1'],'TempGenMean-1': datos_tmenos1['TempGenMean-1'], 'PotMean-1': datos_tmenos1['PotMean-1'], 'VelRotorMean-1': datos_tmenos1['VelRotorMean-1'],'TempEjeLento_1': datos_t['TempEjeLento_1']})


def normalizado(df,a1,b1,a2,b2,a6,b6,a10,b10,a12,b12,a14,b14,a16,b16,a17,b17):
        datos_norm=pd.DataFrame({"date_time": df['date_time']})
        datos_norm["TempEjeLento_1-1"] = (df['TempEjeLento_1-1']-b1)/(a1-b1)
        datos_norm["TempAmbMean-1"]= (df['TempAmbMean-1']-b2)/(a2-b2)
        datos_norm["TempRodamMultipMean-1"] = (df['TempRodamMultipMean-1']-b6)/(a6-b6)
        #datos_norm["TempCojLAMean-1"] = (df['TempCojLAMean-1']-b8)/(a8-b8) 
        datos_norm["TempCojLOAMean-1"] = (df['TempCojLOAMean-1']-b10)/(a10-b10) 
        datos_norm["TempGenMean-1"] = (df['TempGenMean-1']-b12)/(a12-b12)
        datos_norm["PotMean-1"] = (df['PotMean-1']-b14)/(a14-b14) 
        datos_norm["VelRotorMean-1"] = (df['VelRotorMean-1']-b16)/(a16-b16)
        datos_norm["TempEjeLento_1"] = (df['TempEjeLento_1']-b17)/(a17-b17)
        #datos_norm['date_time'] = df['date_time']

        return datos_norm


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


def datosstack(data):
    
        X1 = array(data['TempAmbMean-1'])
        X2 = array(data['TempRodamMultipMean-1'])
        #X3 = array(data['TempCojLAMean-1'])
        X3 = array(data['TempCojLOAMean-1'])
        X4 = array(data['TempGenMean-1'])
        X5 = array(data['PotMean-1'])
        X6 = array(data['VelRotorMean-1'])
        X7 = array(data['TempEjeLento_1-1'])
        X8 = array(data['TempEjeLento_1'])

        X1 = X1.reshape((len(X1), 1))
        X2 = X2.reshape((len(X2), 1))
        X3 = X3.reshape((len(X3), 1))
        X4 = X4.reshape((len(X4), 1))
        X5 = X5.reshape((len(X5), 1))
        X6 = X6.reshape((len(X6), 1))
        X7 = X7.reshape((len(X7), 1))
        X8 = X8.reshape((len(X8), 1))
        #X9 = X8.reshape((len(X9), 1))
        
        dataset = hstack((X1, X2, X3, X4, X5, X6, X7, X8))            

        return dataset 

dataset_train=datosstack(training)
dataset_validation=datosstack(validation)
dataset_test=datosstack(test)


def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		#end_ix = n_steps*i + n_steps
		end_ix = i + n_steps
		#start_ix = end_ix - n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

n_steps = 144
xtrain, ytrain = split_sequences(dataset_train, n_steps)
xvalidation, yvalidation = split_sequences(dataset_validation, n_steps)
xtest, ytest = split_sequences(dataset_test, n_steps)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    #x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = inputs
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs 
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)


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

