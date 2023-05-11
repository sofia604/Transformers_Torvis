from numpy import array
from numpy import hstack


def interpolate_outliers(df):
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

    return df


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