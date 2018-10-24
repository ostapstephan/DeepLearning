import numpy as np
#used to tune the value for random state gen
k = np.random.uniform(-100,100,[10000])/100
r = np.round(k)*1.33
print(np.mean(abs(r)))