import tensorflow as tf
import numpy as np
import math
Size=66
H=1e-1
YitaTotal=np.zeros(Size)
ReTotal=np.zeros(Size)
WeightMatrix1=np.zeros(Size)
WeightMatrix2=np.zeros(Size)
WeightMatrix3=np.zeros(Size)
DragForce=np.zeros(Size)
DragForceWithYita=np.zeros(Size)
#Load the experimental data
dragblue=np.loadtxt('bluedrag.txt')
yitablue=np.loadtxt('blueyita.txt')
dragpink=np.loadtxt('pinkdrag.txt')
yitapink=np.loadtxt('pinkyita.txt')
dragred=np.loadtxt('reddrag.txt')
yitared=np.loadtxt('redyita.txt')
yitagreen=np.loadtxt('greenyita.txt')
draggreen=np.loadtxt('greendrag.txt')
#Discriminate
LargeGate=1.7
#experimental variables
rho=1.0
fL=2e-4
Eflexible=2.8
Erigid=2000
LRed=3.3
LGreen=2.0
LPink=1.8
LBlue=5.2

for i in range(0,len(YitaTotal)):
    if i<21:
        YitaTotal[i]=yitagreen[i]
        ReTotal[i]=yitagreen[i]*math.sqrt(rho*Erigid/fL/LGreen*2.0)
        WeightMatrix1[i]=ReTotal[i]
        WeightMatrix2[i]=YitaTotal[i]
    elif i<33:
        YitaTotal[i]=yitapink[i-21]
        ReTotal[i]=yitapink[i-21]*math.sqrt(rho*Eflexible/fL/LPink*2.0)
        WeightMatrix1[i]=ReTotal[i]
        WeightMatrix2[i]=YitaTotal[i]
    elif i<51:
        YitaTotal[i]=yitared[i-33]
        ReTotal[i]=yitared[i-33]*math.sqrt(rho*Eflexible/fL/LRed*2.0)
        WeightMatrix1[i]=ReTotal[i]
        WeightMatrix2[i]=YitaTotal[i]
    else:
        YitaTotal[i]=yitablue[i-51]
        ReTotal[i]=yitablue[i-51]*math.sqrt(rho*Eflexible/fL/LBlue*2.0)
        WeightMatrix1[i]=ReTotal[i]
        WeightMatrix2[i]=YitaTotal[i]
for i in range(0,len(DragForceWithYita)):
    if i<21:
        DragForceWithYita[i]=draggreen[i]
    elif i<33:
        DragForceWithYita[i]=dragpink[i-21]
    elif i<51:
        DragForceWithYita[i]=dragred[i-33]
    else:
        DragForceWithYita[i]=dragblue[i-51]
        
for i in range(0,len(DragForce)):
        DragForce[i]=DragForceWithYita[i]/YitaTotal[i]/YitaTotal[i]

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
Gamaminus1Number=0
GamaLarge1Number=0
for i in range(0,len(YitaTotal)):
    if(YitaTotal[i]<LargeGate):
        Gamaminus1Number=Gamaminus1Number+1
    else:
        GamaLarge1Number=GamaLarge1Number+1
Gama1Input1Minus1=np.zeros(Gamaminus1Number)
Gama2Input1Minus1=np.zeros(Gamaminus1Number)
Gama3Input1Minus1=np.zeros(Gamaminus1Number)
Gama1Input2Minus1=np.zeros(Gamaminus1Number)
Gama2Input2Minus1=np.zeros(Gamaminus1Number)
Gama3Input2Minus1=np.zeros(Gamaminus1Number)
Gama1Input1Large1=np.zeros(GamaLarge1Number)
Gama2Input1Large1=np.zeros(GamaLarge1Number)
Gama3Input1Large1=np.zeros(GamaLarge1Number)
Gama1Input2Large1=np.zeros(GamaLarge1Number)
Gama2Input2Large1=np.zeros(GamaLarge1Number)
Gama3Input2Large1=np.zeros(GamaLarge1Number)
Reminus1=np.zeros(Gamaminus1Number)
ReLarge1=np.zeros(GamaLarge1Number)
Yitaminus1=np.zeros(Gamaminus1Number)
YitaLarge1=np.zeros(GamaLarge1Number)
WeightMatrix1Minus1=np.zeros(Gamaminus1Number)
WeightMatrix1Large1=np.zeros(GamaLarge1Number)
WeightMatrix2Minus1=np.zeros(Gamaminus1Number)
WeightMatrix2Large1=np.zeros(GamaLarge1Number)
DragForceminus1=np.zeros(Gamaminus1Number)
DragForceLarge1=np.zeros(GamaLarge1Number)
DragForceWithYitaminus1=np.zeros(Gamaminus1Number)
DragForceWithYitaLarge1=np.zeros(GamaLarge1Number)
j=0
k=0
for i in range(0,len(YitaTotal)):
    if(YitaTotal[i]<LargeGate):
        Reminus1[j]=ReTotal[i]
        Yitaminus1[j]=YitaTotal[i]
        DragForceminus1[j]=DragForce[i]
        DragForceWithYitaminus1[j]=DragForceWithYita[i]
        WeightMatrix1Minus1[j]=WeightMatrix1[i]
        WeightMatrix2Minus1[j]=WeightMatrix2[i]
        Gama1Input1Minus1[j]=WeightMatrix1[i]*10**(H*(w[0,0]+w[0,4]+w[0,2]-w[0,1]))
        Gama2Input1Minus1[j]=WeightMatrix1[i]*10**(H*(w[1,0]+w[1,4]+w[1,2]-w[1,1]))
        Gama3Input1Minus1[j]=WeightMatrix1[i]*10**(H*(w[2,0]+w[2,4]+w[2,2]-w[2,1]))
        Gama1Input2Minus1[j]=WeightMatrix2[i]*10**(H*(w[0,0]/2.0+w[0,3]*1.0/2.0+w[0,4]*2.0/2.0+w[0,2]*3.0/2.0-w[0,5]/2.0))
        Gama2Input2Minus1[j]=WeightMatrix2[i]*10**(H*(w[1,0]/2.0+w[1,3]*1.0/2.0+w[1,4]*2.0/2.0+w[1,2]*3.0/2.0-w[1,5]/2.0))
        Gama3Input2Minus1[j]=WeightMatrix2[i]*10**(H*(w[2,0]/2.0+w[2,3]*1.0/2.0+w[2,4]*2.0/2.0+w[2,2]*3.0/2.0-w[2,5]/2.0))
        j=j+1
    else:
        ReLarge1[k]=ReTotal[i]
        YitaLarge1[k]=YitaTotal[i]
        DragForceLarge1[k]=DragForce[i]
        DragForceWithYitaLarge1[k]=DragForceWithYita[i]
        WeightMatrix1Large1[k]=WeightMatrix1[i]
        WeightMatrix2Large1[k]=WeightMatrix2[i]
        Gama1Input1Large1[k]=WeightMatrix1[i]*10**(H*(w[0,0]+w[0,4]+w[0,2]-w[0,1]))
        Gama2Input1Large1[k]=WeightMatrix1[i]*10**(H*(w[1,0]+w[1,4]+w[1,2]-w[1,1]))
        Gama3Input1Large1[k]=WeightMatrix1[i]*10**(H*(w[2,0]+w[2,4]+w[2,2]-w[2,1]))
        Gama1Input2Large1[k]=WeightMatrix2[i]*10**(H*(w[0,0]/2.0+w[0,3]*1.0/2.0+w[0,4]*2.0/2.0+w[0,2]*3.0/2.0-w[0,5]/2.0))
        Gama2Input2Large1[k]=WeightMatrix2[i]*10**(H*(w[1,0]/2.0+w[1,3]*1.0/2.0+w[1,4]*2.0/2.0+w[1,2]*3.0/2.0-w[1,5]/2.0))
        Gama3Input2Large1[k]=WeightMatrix2[i]*10**(H*(w[2,0]/2.0+w[2,3]*1.0/2.0+w[2,4]*2.0/2.0+w[2,2]*3.0/2.0-w[2,5]/2.0))
        k=k+1 
    
x1_train=(WeightMatrix1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_train=(WeightMatrix2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x_train=np.array([x1_train,x2_train])
x_train=x_train.T
y_train=DragForceLarge1.copy()

x1_test=(Gama1Input1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_test=(Gama1Input2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x_predict1=np.array([x1_test,x2_test])
x_predict1=x_predict1.T
y_predict1=np.zeros(GamaLarge1Number)

x1_test2=(Gama2Input1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_test2=(Gama2Input2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x_predict2=np.array([x1_test2,x2_test2])
x_predict2=x_predict2.T
y_predict2=np.zeros(GamaLarge1Number)

x1_test3=(Gama3Input1Large1.copy()-np.min(WeightMatrix1Large1))/(np.max(WeightMatrix1Large1)-np.min(WeightMatrix1Large1))
x2_test3=(Gama3Input2Large1.copy()-np.min(WeightMatrix2Large1))/(np.max(WeightMatrix2Large1)-np.min(WeightMatrix2Large1))
x_predict3=np.array([x1_test3,x2_test3])
x_predict3=x_predict3.T
y_predict3=np.zeros(GamaLarge1Number)

model1=tf.keras.models.Sequential()
model1=tf.keras.models.load_model('modelLarge1little')
y_predict1=model1.predict(x_predict1)
y_predict2=model1.predict(x_predict2)
y_predict3=model1.predict(x_predict3)

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
print(e_vals,e_vecs)
print(w[0,:]*e_vecs[0,0]+w[1,:]*e_vecs[1,0]+w[2,:]*e_vecs[2,0])
print(w[0,:]*e_vecs[0,1]+w[1,:]*e_vecs[1,1]+w[2,:]*e_vecs[2,1])
print(w[0,:]*e_vecs[0,2]+w[1,:]*e_vecs[1,2]+w[2,:]*e_vecs[2,2])

np.savetxt("Large1Predict1",np.hstack((x_predict1,y_predict1)))
np.savetxt("Large1Predict2",np.hstack((x_predict2,y_predict2)))
np.savetxt("Large1Predict3",np.hstack((x_predict3,y_predict3)))
np.savetxt("Large1Original",np.hstack((x_train,y_train.reshape([len(y_train),1]))))
np.savetxt("Large1testfy",np.hstack((x_train,model1.predict(x_train))))
