import os
import numpy as np
seed = 123
rng = np.random.default_rng(seed)

lr = [0.1, 0.01, 0.001, 0.0001]
batch_size = [8, 16, 32, 64]
epochs = [20, 35, 50, 65, 80]

lr = rng.choice(lr, size = 10, replace = True)
batch_size = rng.choice(batch_size, size = 10, replace = True)
epochs = rng.choice(epochs, size = 10, replace = True)

for i in range(10):
    os.system('python run.py --lr='+str(lr[i])+' --batch_size='+str(batch_size[i])+' --epochs='+str(epochs[i]))
    os.system('python metrics.py --path=./predictions/'+str(lr[i])+'_'+str(batch_size[i])+'_'+str(epochs[i])+'.csv --threshold='+str(0.5))