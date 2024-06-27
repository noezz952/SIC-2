import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("ai4i2020.csv")


data2 = data.drop(data.columns[[0,1,-1,-2,-3,-4,-5]], axis=1)

#ganti type L, M, H ke 0.5, 0.3, 0.2
d = {"H":0.2,'M':0.3,"L":0.5}
data2["Type"]= data2["Type"].map(d)

# print(data2)

tgt = data2["Machine failure"].values.reshape(-1,1) #target
ftr = data2.drop(["Machine failure"], axis=1).values.reshape(-1,6) #feature

#pisahkan data test dan data model
xtrain, xtest, ytrain, ytest = train_test_split(ftr,tgt, train_size=0.8) #dari data dibagi 0.8 untuk train dan sisa 0.2 ntuk test

#pemodelan
model = LogisticRegression()
model.fit(xtrain,ytrain)


#prediksi dan tes akurasi
pred = model.predict(xtest)
akura = accuracy_score (ytest, pred)
print("akurasi data : ", akura*100,"%") #akurasi dari hasil model 5 fitur dan target (MF)

typ = input("masukkan type : ")
at = int(input("Air temperature [K] : "))
pt = int(input("Process temperature [K] : "))
rs = int(input("Rational speed [rpm] : "))  
tq = int(input("Torque [Nm] : "))
tw = int(input("Tool wear [min] : "))

data_input = [typ,at,pt,rs,tq,tw]
if typ == "M":
    data_input[0] = float(0.3)
elif typ == "H":
    data_input[0] = float(0.2)
else :
    data_input[0] = float(0.5)

return_data = np.array([data_input])

pred = model.predict(return_data)
print("Prediksi Failure : ", pred)