import h5py
import numpy as np
import pylab as pl

h5f = h5py.File('logs/log.h5','r')

X1 = h5f['g'][:]

h5f.close()

F = pl.figure(figsize=(10,8))
f = F.add_subplot(111)

res = 21
[y1,x1] = np.histogram(X1, bins=np.logspace(np.log10(1),np.log10(100000), res))

z1 = np.where(y1==0)
y1 = np.delete(y1,z1)
x1 = np.delete(x1,z1)

f.plot(x1[:-1],y1,'.-',color=(0,0,1),markersize=18)

f.set_xlim([10,10**7])
f.set_aspect(1)
f.set_xscale("log")
f.set_yscale("log")
f.set_xlabel("generations",fontsize=20)
f.set_ylabel("evolutions",fontsize=20)

pl.show()
#F.savefig("figure.pdf",dpi=300)
