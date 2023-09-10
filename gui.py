#GUI section
import joblib
from wine_quality_prediction import pca, st
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# pca=PCA(n_components=0.90)
# st=StandardScaler()


model =joblib.load('wine_quality_prediction.joblib')
# # Prediction on New Data
import pandas as pd 
new_data=pd.DataFrame({
 'fixed acidity':7.3,
 'volatile acidity':0.65,
 'citric acid':0.00,
 'residual sugar':1.2,
 'chlorides':0.065,
 'free sulfur dioxide':15.0,
 'total sulfur dioxide':21.0,
 'density':0.9946,
 'pH':3.39,
 'sulphates':0.47,
 'alcohol':10.0
},index=[0])
new_data
test=pca.transform(st.transform(new_data))
p=model.predict(test)
if p[0]==1:
 print ("good quality wine")
else:
 print("bad quality wine")
# # GUI
from tkinter import *
from sklearn.preprocessing import StandardScaler
import joblib
def show_entry_fields():
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    p7=float(e7.get())
    p8=float(e8.get())
    p9=float(e9.get())
    p10=float(e10.get())
    p11=float(e11.get())
    model=joblib.load('wine_quality_prediction.joblib')
    result=model.predict(pca.transform(st.transform([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]])))
    Label(master,text="quality").grid(row=12)
    Label(master,text=result).grid(row=13)
    if result[0] == 0:
        Label(master, text="Bad Quality Wine").grid(row=31)
    else:
        Label(master, text="Good Quality Wine").grid(row=31)

        
master=Tk()
master.title("wine quality prediction")
label=Label(master,text="wine quality prediction",bg="black",fg="white").grid(row=0,columnspan=2)
Label(master,text="fixed acidity").grid(row=1)
Label(master,text="volatile acidity").grid(row=2)
Label(master,text="citric acid").grid(row=3)
Label(master,text="residual sugar").grid(row=4)
Label(master,text="chlorides").grid(row=5)
Label(master,text="free sulfur dioxide").grid(row=6)
Label(master,text="total sulfur dioxide").grid(row=7)
Label(master,text="density").grid(row=7)
Label(master,text="pH").grid(row=8)
Label(master,text="sulphates").grid(row=9)
Label(master,text="alcohol").grid(row=10)
Label(master,text="density").grid(row=11)
e1=Entry(master)
e2=Entry(master)
e3=Entry(master)
e4=Entry(master)
e5=Entry(master)
e6=Entry(master)
e7=Entry(master)
e8=Entry(master)
e9=Entry(master)
e10=Entry(master)
e11=Entry(master)
e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)
e8.grid(row=8,column=1)
e9.grid(row=9,column=1)
e10.grid(row=10,column=1)
e11.grid(row=11,column=1)
Button(master,text='Predict',command=show_entry_fields).grid()
mainloop()