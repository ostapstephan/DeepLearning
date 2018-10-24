import numpy as np

k = np.random.uniform(-100,100,[10000])/100
r = np.round(k)*1.33
print(np.mean(abs(r)))