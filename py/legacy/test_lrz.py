from _lrz import *
from _errlog import *
import os

home = os.path.expanduser("~")
fname = os.path.join(home, "data", "cesm_dim3-3600x1800", "CLDHGH.f32")
d = np.fromfile(fname, dtype="u4")


d1 = d.reshape((6480000))
d2 = d.reshape((1800, 3600))
d3 = d.reshape((180, 360, 100))


e1 = np.zeros_like(d1)
xd1 = np.zeros_like(d1)
e2 = np.zeros_like(d2)
xd2 = np.zeros_like(d2)
e3 = np.zeros_like(d3)
xd3 = np.zeros_like(d3)

PszApprox.succeeding_lorenzo_predict(d1, e1)
PszApprox.reverse_lorenzo_predict(e1, xd1)

PszApprox.succeeding_lorenzo_predict(d2, e2)
PszApprox.reverse_lorenzo_predict(e2, xd2)

PszApprox.succeeding_lorenzo_predict(d3, e3)
PszApprox.reverse_lorenzo_predict(e3, xd3)

print((d1 == xd1).all())
print((d2 == xd2).all())
print((d3 == xd3).all())
