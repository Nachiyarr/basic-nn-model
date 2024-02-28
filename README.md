# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/Nachiyarr/basic-nn-model/assets/113497340/0e3fab66-e41d-4b3c-9fce-d141f964e8ef)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:ALAGU NACHIYAR K
### Register Number:21222224006
```

import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('dlex-1').sheet1


rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'int'})
df = df.astype({'output':'int'})
df.head()



x = df[['input']].values
y = df[['output']].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.54,random_state = 54)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train1 = Scaler.transform(x_train)
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs = 1000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
x_n1 = [[30]]
x_n1_1 = Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)


```
## Dataset Information

![image](https://github.com/Nachiyarr/basic-nn-model/assets/113497340/bbf5de56-8b6e-43fa-bf26-47c47d21080e)


## OUTPUT
![image](https://github.com/Nachiyarr/basic-nn-model/assets/113497340/3ba204b2-2a6c-4dbb-a660-aa06b14bafac)




### Test Data Root Mean Squared Error

![image](https://github.com/Nachiyarr/basic-nn-model/assets/113497340/5f17beb9-af7b-48ec-a42b-2a0cc1380d2e)


### New Sample Data Prediction

![image](https://github.com/Nachiyarr/basic-nn-model/assets/113497340/3b0305a7-2398-4d23-bd13-425db492ef83)


## RESULT
Thus the Process of developing a neural network regression model for the created dataset is successfully executed.

Include your result here
