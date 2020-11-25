import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
df=pd.read_csv('data.txt', index_col=[0])
df=data[['pm2.5','TEMP','DEWP','PRES','Ir','Iws','Is']]

plt.plot(df['pm2.5'])
plt.show();
plt.plot(df['TEMP'])
plt.show();
plt.plot(df['DEWP'])
plt.show();
plt.plot(df['PRES'])
plt.show();

df['pm2.5']=df['pm2.5']/1000
df['PRES']=df['PRES']/1000
df.shape

def subsample_sequence(df, length):
    
    last_possible = df.shape[0] - length
    
    random_start = np.random.randint(0, last_possible)
    df_sample = df[random_start: random_start+length]
    
    return df_sample

df_subsample = subsample_sequence(df, 10)
df_subsample.shape

def fillna(X, df_mean):
    # Replace with NaN of the other hours.
    na = X.mean()
    
    # If the other hours are also nans, then replace with mean value of the dataframe
    na.fillna(df_mean)
    return na

def split_subsample_sequence(df, length, df_mean=None):
    # Little trick to improve the time
    if df_mean is None:
        df_mean = df.mean()
              
    df_subsample = subsample_sequence(df, length)
    y_sample = df_subsample.iloc[df_subsample.shape[0]-1]['pm2.5']
    
    if y_sample != y_sample: # A value is not equal to itself only for NaN. So it will be True if y_sample is nan
        X_sample, y_sample = split_subsample_sequence(df, length, df_mean)
        return np.array(X_sample), np.array(y_sample)
    
    X_sample = df_subsample[0:df_subsample.shape[0]-1]
    X_sample = X_sample.fillna(fillna(X_sample, df_mean))
    X_sample = X_sample.values
    
    return np.array(X_sample), np.array(y_sample)

def get_X_y(df, number_of_sequences, length):
    ### YOUR CODE HERE
    X=[]
    y=[]
    
    for i in range(number_of_sequences):
        xi, yi= split_subsample_sequence(df, length)
        X.append(xi)
        y.append(yi)
        
    X=np.array(X)
    y=np.array(y)
    
    return X, y

X, y = get_X_y(df, 100, 21)
X.shape
length=int(0.8*df.shape[0])
df_train=df[0:length]
df_test=df[length:]
X_train, y_train=get_X_y(df_train, 1000, 51)
X_test, y_test=get_X_y(df_test, 200, 51)

def init_model():
    model=Sequential()
    model.add(layers.LSTM(units=20, activation='tanh'))
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mae'])
    return model
model=init_model()
es=EarlyStopping(patience=2,monitor='val_loss',mode='min',restore_best_weights=True)
model.fit(X_train, y_train,
          batch_size=16,
          epochs=10,
          validation_split=0.2,
          callbacks=[es],
          verbose=1)

result=model.evaluate(X_test, y_test)
result[1]*100

y_pred=np.mean(y_train)
bench_resnp=np.mean(np.abs(y_test-y_pred))
bench_resnp*100
