import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

df=pd.read_csv('/content/letter-recognition.csv')

X=df.drop('letter',axis=1)
y=df['letter']
y = y.apply(lambda x: ord(x) - ord('A'))
y=to_categorical(y,num_classes=26)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=Sequential([
    Dense(128,activation='relu'),
    Dropout(0.3),
    Dense(64,activation='relu'),
    Dropout(0.3),
    Dense(26,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=20,batch_size=32,validation_split=0.2,verbose=1)
test_loss,test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy)
predictions=model.predict(X_test)
print(predictions)
y_pred=predictions.argmax(axis=1)
y_true=y_test.argmax(axis=1)
print(classification_report(y_true,y_pred,target_names=[chr(i) for i in range(ord('A'),ord('Z')+1)]))