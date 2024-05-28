import numpy as np
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(-10,10))

a = np.array([1,2,4,20,-20,5,0]).reshape(-1,1)
b = np.array([1,2,4,30,-50,25,0]).reshape(-1,1)

print(a)
print(b)

new_a = scaler.fit_transform(a)
#new_b = scaler.fit_transform(b)

print("NEw a", new_a)
#print("NEW b", new_b)

new_a2 = scaler.transform(a)
new_b2 = scaler.transform(b)

print("new A2", new_a2)
print("new B2", new_b2)