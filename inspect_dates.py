import numpy as np,sys
z=np.load(sys.argv[1],allow_pickle=False)
print(np.unique(z['feature_date'])[:10].tolist())
