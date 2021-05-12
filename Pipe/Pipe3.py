import tensorflow as tf
import numpy as np
import math

def f(x,Ref,Roughnessf):
    return x**(0-0.5)+2.0*math.log10(Roughnessf/3.7+2.51/Ref*x**(0-0.5))
def fd(x,Ref,Roughnessf):
    return 0-0.5*(x)**(0-1.5)+2.0*(2.51/Ref*(0-0.5)*x**(0-1.5))/(Roughnessf/3.7+2.51/Ref*x**(0-0.5))/math.log(10)
def newtonMethod(assum,Ref,Roughnessf):
    if Ref<3000:
        return 64.0/Ref
    else:
        x0 = assum
        x1 = 0
        while abs(f(x0,Ref,Roughnessf))>1e-12:
            a = f(x0,Ref,Roughnessf)
            b = fd(x0,Ref,Roughnessf)
            x1 = x0-a/b
            x0=x1
        return x0

np.random.seed(60)
SIEZZ=1000
H=2e-2

#VelocityLower=2.5e-2
#VelocityUpper=3.0e-2
#VelocityLower=2.0
#VelocityUpper=4.0
VelocityLower=500
VelocityUpper=700
VelocityTotal=np.random.uniform(VelocityLower,VelocityUpper,size=(SIEZZ))
DensityLower=0.1
DensityUpper=0.14
DensityTotal=np.random.uniform(DensityLower,DensityUpper,size=(SIEZZ))
ViscosityLower=1e-6
ViscosityUpper=1e-5
ViscosityTotal=np.random.uniform(ViscosityLower,ViscosityUpper,size=(SIEZZ))
DiameterLower=0.5
#DiameterUpper=0.8
DiameterUpper=1.0
DiameterTotal=np.random.uniform(DiameterLower,DiameterUpper,size=(SIEZZ))
#RoughnessLower=3e-5
#RoughnessUpper=8e-5
#RoughnessLower=5e-4
#RoughnessUpper=2e-3
RoughnessLower=1e-2
RoughnessUpper=4e-2
RoughnessTotal=np.random.uniform(RoughnessLower,RoughnessUpper,size=(SIEZZ))

WeightMatrix1=np.zeros(SIEZZ)
WeightMatrix2=np.zeros(SIEZZ)
PressureDrop=np.zeros(SIEZZ)

for i in range(0,SIEZZ):
    WeightMatrix1[i]=DensityTotal[i]*VelocityTotal[i]*DiameterTotal[i]/ViscosityTotal[i]
    WeightMatrix2[i]=RoughnessTotal[i]/DiameterTotal[i]
    PressureDrop[i]=newtonMethod(0.001,WeightMatrix1[i],WeightMatrix2[i])

w=np.zeros((2,5))
#rho mu D eta V
w[0][0]=0.5
w[0][1]=0-0.5
w[0][2]=0.5
w[0][4]=0.5
w[1][0]=math.sqrt(7)/14.0*1.0
w[1][1]=math.sqrt(7)/14.0*(0.0-1.0)
w[1][2]=math.sqrt(7)/14.0*(0.0-3.0)
w[1][3]=math.sqrt(7)/14.0*4.0
w[1][4]=math.sqrt(7)/14.0*1.0

Gama1Input1=np.random.uniform(0,0,size=(SIEZZ))
Gama2Input1=np.random.uniform(0,0,size=(SIEZZ))
Gama1Input2=np.random.uniform(0,0,size=(SIEZZ))
Gama2Input2=np.random.uniform(0,0,size=(SIEZZ))
y_real1=np.random.uniform(0,0,size=(SIEZZ))
y_real2=np.random.uniform(0,0,size=(SIEZZ))
for i in range(0,SIEZZ):
     Gama1Input1[i]=WeightMatrix1[i]*10**(H*(w[0,0]-w[0,1]+w[0,2]+w[0,4]))
     Gama2Input1[i]=WeightMatrix1[i]*10**(H*(w[1,0]-w[1,1]+w[1,2]+w[1,4]))
     Gama1Input2[i]=WeightMatrix2[i]*10**(H*(w[0,3]-w[0,2]))
     Gama2Input2[i]=WeightMatrix2[i]*10**(H*(w[1,3]-w[1,2]))
     y_real1[i]=newtonMethod(0.001,Gama1Input1[i],Gama1Input2[i])
     y_real2[i]=newtonMethod(0.001,Gama2Input1[i],Gama2Input2[i])
REUPPER=DensityUpper*VelocityUpper*DiameterUpper/ViscosityLower
REDOWN=DensityLower*VelocityLower*DiameterLower/ViscosityUpper
ROUGHNESSDimensionallessUP=RoughnessUpper/DiameterLower
ROUGHNESSDimensionallessDOWN=RoughnessLower/DiameterUpper

x1_train=(WeightMatrix1.copy()-REDOWN)/(REUPPER-REDOWN)
x2_train=(WeightMatrix2.copy()-ROUGHNESSDimensionallessDOWN)/(ROUGHNESSDimensionallessUP-ROUGHNESSDimensionallessDOWN)
x_train=np.array([x1_train,x2_train])
x_train=x_train.T
y_train=PressureDrop.copy()

x1_test=(Gama1Input1.copy()-REDOWN)/(REUPPER-REDOWN)
x2_test=(Gama1Input2.copy()-ROUGHNESSDimensionallessDOWN)/(ROUGHNESSDimensionallessUP-ROUGHNESSDimensionallessDOWN)
x_predict1=np.array([x1_test,x2_test])
x_predict1=x_predict1.T
y_predict1=np.zeros(SIEZZ)  

x1_test2=(Gama2Input1.copy()-REDOWN)/(REUPPER-REDOWN)
x2_test2=(Gama2Input2.copy()-ROUGHNESSDimensionallessDOWN)/(ROUGHNESSDimensionallessUP-ROUGHNESSDimensionallessDOWN)
x_predict2=np.array([x1_test2,x2_test2])
x_predict2=x_predict2.T
y_predict2=np.zeros(SIEZZ) 

model1=tf.keras.models.Sequential()
model1=tf.keras.models.load_model('Pipe3')
y_predict1=model1.predict(x_predict1)
y_predict2=model1.predict(x_predict2)
error=0.0
for i in range(0,SIEZZ):
    error=error+abs(y_predict1[i]-y_real1[i])
    error=error+abs(y_predict2[i]-y_real2[i])
print(error)
C11=0.0
C12=0.0
C22=0.0
for i in range(0,SIEZZ):
    C11=C11+((y_predict1[i]/1.0-PressureDrop[i]/1.0)/H)**2.0
    C12=C12+(y_predict1[i]/1.0-PressureDrop[i]/1.0)/H*(y_predict2[i]/1.0-PressureDrop[i]/1.0)/H
    C22=C22+((y_predict2[i]/1.0-PressureDrop[i]/1.0)/H)**2.0
C11=C11/(SIEZZ)
C12=C12/(SIEZZ)
C22=C22/(SIEZZ)
AssambleMat=np.mat(np.array([[C11,C12],[C12,C22]]))
e_vals,e_vecs = np.linalg.eig(AssambleMat)
print(e_vals,e_vecs)
print(w[0,:]*e_vecs[0,0]+w[1,:]*e_vecs[1,0])
print(w[0,:]*e_vecs[0,1]+w[1,:]*e_vecs[1,1])
