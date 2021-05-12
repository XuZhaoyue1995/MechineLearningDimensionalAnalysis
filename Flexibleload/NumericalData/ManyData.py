import tensorflow as tf
import numpy as np
import math

np.random.seed(60)
#Total Number of numerical case
Size=1000
#Difference step          
H=2e-2
#Variables between upbound and lowbound! Attention, in order to keep Yita
#larger than 1, we define Yita rather than Velocity.
YitaTotal=np.random.uniform(1.0,20.0,size=(Size))
Rho=np.random.uniform(0.8,1.2,size=(Size))
FL=np.random.uniform(1e-4,4e-4,size=(Size))
Eflexible=np.random.uniform(1.0,4.0,size=(Size))
Lline=np.random.uniform(4.0,8.0,size=(Size))
#Other Variables and three nondimentional matrix.
Velocity=np.zeros(Size)
ReTotal=np.zeros(Size)
WeightMatrix1=np.zeros(Size)
WeightMatrix2=np.zeros(Size)
WeightMatrix3=np.zeros(Size)
DragForce=np.zeros(Size)
DragForceWithYita=np.zeros(Size)

#The equation ratio from Alben's article.
Ratio=1.87
for i in range(0,len(YitaTotal)):
    ReTotal[i]=YitaTotal[i]*math.sqrt(Rho[i]*Eflexible[i]/FL[i]/Lline[i]*2.0)
    Velocity[i]=YitaTotal[i]*math.sqrt(Eflexible[i]/Rho[i]/FL[i]/Lline[i]**3.0*2.0)
    WeightMatrix1[i]=YitaTotal[i]**2.0/FL[i]*Lline[i]*2.0
    WeightMatrix2[i]=ReTotal[i]/Lline[i]*FL[i]
    WeightMatrix3[i]=ReTotal[i]
        
for i in range(0,len(DragForce)):
        DragForceWithYita[i]=YitaTotal[i]**(4.0/3.0)*Ratio+np.random.normal(loc =0.0 , scale= 0.5)*0.001
        DragForce[i]=(YitaTotal[i]**(0-2.0/3.0))*Ratio+np.random.normal(loc =0.0 , scale= 0.5)*0.001
#We define w the same as W matrix in my article.
w=np.zeros((3,6))
w[0,2]=math.sqrt(2)/2.0
w[0,3]=0-math.sqrt(2)/2.0
w[1,0]=math.sqrt(14)/7.0*1.0
w[1,1]=math.sqrt(14)/7.0*(0.0-1.0)
w[1,2]=math.sqrt(14)/14.0
w[1,3]=math.sqrt(14)/14.0
w[1,4]=math.sqrt(14)/7.0*1.0
w[1,5]=0
w[2,0]=0-math.sqrt(21.0)/28.0
w[2,1]=5.0*math.sqrt(21.0)/42.0
w[2,2]=3.0*math.sqrt(21.0)/28.0
w[2,3]=3.0*math.sqrt(21.0)/28.0
w[2,4]=math.sqrt(21.0)/21.0
w[2,5]=0-math.sqrt(21.0)/12.0

GamaLarge1Number=Size
#Define the different matrix
Gama1Input1Large1=np.zeros(GamaLarge1Number)
Gama2Input1Large1=np.zeros(GamaLarge1Number)
Gama3Input1Large1=np.zeros(GamaLarge1Number)
Gama1Input2Large1=np.zeros(GamaLarge1Number)
Gama2Input2Large1=np.zeros(GamaLarge1Number)
Gama3Input2Large1=np.zeros(GamaLarge1Number)
Gama1Input3Large1=np.zeros(GamaLarge1Number)
Gama2Input3Large1=np.zeros(GamaLarge1Number)
Gama3Input3Large1=np.zeros(GamaLarge1Number)
ReLarge1=np.zeros(GamaLarge1Number)
YitaLarge1=np.zeros(GamaLarge1Number)
WeightMatrix1Large1=np.zeros(GamaLarge1Number)
WeightMatrix2Large1=np.zeros(GamaLarge1Number)
WeightMatrix3Large1=np.zeros(GamaLarge1Number)
DragForceLarge1=np.zeros(GamaLarge1Number)
DragForceWithYitaLarge1=np.zeros(GamaLarge1Number)

k=0
for i in range(0,len(YitaTotal)):
    ReLarge1[k]=ReTotal[i]
    YitaLarge1[k]=YitaTotal[i]
    DragForceLarge1[k]=DragForce[i]
    DragForceWithYitaLarge1[k]=DragForceWithYita[i]
    WeightMatrix1Large1[k]=WeightMatrix1[i]
    WeightMatrix2Large1[k]=WeightMatrix2[i]
    WeightMatrix3Large1[k]=WeightMatrix3[i]
    Gama1Input1Large1[k]=WeightMatrix1[i]*10**(H*(w[0,0]+w[0,3]*0.0+w[0,4]*2.0+w[0,2]*4.0-w[0,5]))
    Gama2Input1Large1[k]=WeightMatrix1[i]*10**(H*(w[1,0]+w[1,3]*0.0+w[1,4]*2.0+w[1,2]*4.0-w[1,5]))
    Gama3Input1Large1[k]=WeightMatrix1[i]*10**(H*(w[2,0]+w[2,3]*0.0+w[2,4]*2.0+w[2,2]*4.0-w[2,5]))
    Gama1Input2Large1[k]=WeightMatrix2[i]*10**(H*(w[0,0]+w[0,4]+w[0,3]-w[0,1]))
    Gama2Input2Large1[k]=WeightMatrix2[i]*10**(H*(w[1,0]+w[1,4]+w[1,3]-w[1,1]))
    Gama3Input2Large1[k]=WeightMatrix2[i]*10**(H*(w[2,0]+w[2,4]+w[2,3]-w[2,1]))
    Gama1Input3Large1[k]=WeightMatrix3[i]*10**(H*(w[0,0]+w[0,4]+w[0,2]-w[0,1]))
    Gama2Input3Large1[k]=WeightMatrix3[i]*10**(H*(w[1,0]+w[1,4]+w[1,2]-w[1,1]))
    Gama3Input3Large1[k]=WeightMatrix3[i]*10**(H*(w[2,0]+w[2,4]+w[2,2]-w[2,1]))
    k=k+1 
#Generate the training set   
x1_train=(WeightMatrix1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_train=(WeightMatrix2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x3_train=(WeightMatrix3Large1.copy()-np.min(WeightMatrix3Large1))/(np.max(WeightMatrix3Large1)-np.min(WeightMatrix3Large1))
x_train=np.array([x1_train,x2_train,x3_train])
x_train=x_train.T
y_train=DragForceLarge1.copy()

x1_test=(Gama1Input1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_test=(Gama1Input2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x3_test=(Gama1Input3Large1.copy()-np.min(WeightMatrix3Large1))/(np.max(WeightMatrix3Large1)-np.min(WeightMatrix3Large1))
x_predict1=np.array([x1_test,x2_test,x3_test])
x_predict1=x_predict1.T
y_predict1=np.zeros(GamaLarge1Number)

x1_test2=(Gama2Input1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_test2=(Gama2Input2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x3_test2=(Gama2Input3Large1.copy()-np.min(WeightMatrix3Large1))/(np.max(WeightMatrix3Large1)-np.min(WeightMatrix3Large1))
x_predict2=np.array([x1_test2,x2_test2,x3_test2])
x_predict2=x_predict2.T
y_predict2=np.zeros(GamaLarge1Number)

x1_test3=(Gama3Input1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_test3=(Gama3Input2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x3_test3=(Gama3Input3Large1.copy()-np.min(WeightMatrix3Large1))/(np.max(WeightMatrix3Large1)-np.min(WeightMatrix3Large1))
x_predict3=np.array([x1_test3,x2_test3,x3_test3])
x_predict3=x_predict3.T
y_predict3=np.zeros(GamaLarge1Number)

#load model and predict the data
model1=tf.keras.models.Sequential()
model1=tf.keras.models.load_model('FinalLarge1')
y_predict1=model1.predict(x_predict1)
y_predict2=model1.predict(x_predict2)
y_predict3=model1.predict(x_predict3)
#compute the active subspace
C11=0.0
C12=0.0
C13=0.0
C22=0.0
C11=0.0
C12=0.0
C22=0.0
C13=0.0
C23=0.0
C33=0.0
for i in range(0,GamaLarge1Number):
    C11=C11+((y_predict1[i]-DragForceLarge1[i])/H)**2.0
    C12=C12+(y_predict1[i]-DragForceLarge1[i])/H*(y_predict2[i]-DragForceLarge1[i])/H
    C22=C22+((y_predict2[i]-DragForceLarge1[i])/H)**2.0
    C13=C13+(y_predict1[i]-DragForceLarge1[i])/H*(y_predict3[i]-DragForceLarge1[i])/H
    C23=C23+(y_predict2[i]-DragForceLarge1[i])/H*(y_predict3[i]-DragForceLarge1[i])/H
    C33=C33+((y_predict3[i]-DragForceLarge1[i])/H)**2.0
C11=C11/GamaLarge1Number
C12=C12/GamaLarge1Number
C22=C22/GamaLarge1Number
C13=C13/GamaLarge1Number
C23=C23/GamaLarge1Number
C33=C33/GamaLarge1Number
AssambleMat=np.mat(np.array([[C11,C12,C13],[C12,C22,C23],[C13,C23,C33]]))
e_vals,e_vecs = np.linalg.eig(AssambleMat)
#print eigenvalues and eigenvectors. Print Z in the article.
print(e_vals,e_vecs)
print(w[0,:]*e_vecs[0,0]+w[1,:]*e_vecs[1,0]+w[2,:]*e_vecs[2,0])
print(w[0,:]*e_vecs[0,1]+w[1,:]*e_vecs[1,1]+w[2,:]*e_vecs[2,1])
print(w[0,:]*e_vecs[0,2]+w[1,:]*e_vecs[1,2]+w[2,:]*e_vecs[2,2])
