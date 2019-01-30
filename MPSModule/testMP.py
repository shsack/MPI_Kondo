import numpy as np
import bisect
import copy
import numbers
import msphys
from MPSModule.MatrixProducts import MPS
from MPSModule.MatrixProducts import MPO

a =MPS()
a.setRandomState()
b = MPS()
b.setRandomState()

print(a.conjugate()*a)

#exit()

a = MPS()
a.setRandomState()
b = MPO()
b.setRandomState()
c = a.conjugate() * b
print(c)
d = c * a
a.normalise()
c = a.conjugate()
d = copy.deepcopy(b)
c.makeCanonical()
c = c.conjugate()
d.makeCanonical()
t = np.asanyarray([[[ state1.conjugate() * operator * state2
                     for state2 in (a,c)] for operator in (b,d)] for state1 in (a,c)])
t = t.flatten()
t1,t2 = np.meshgrid(t,t)
print(np.abs(t1-t2 ) / np.abs(t1) < 10**-14) # machine precision errors are ok


a = MPS()
b = MPS()
a.setRandomState()
b.setRandomState()
print((a.conjugate()*b)**2)
c = b*a.conjugate()
print(a.conjugate()*c*b)
c.makeCanonical() # notice the speed difference!
print(a.conjugate()*c*b)
d = a.conjugate()*c
print(d*b)
d.makeCanonical() # something is going wrong here! Any ideas? !!!! State is being normalised
print(d*b)

a = MPS()
a.setRandomState()
b = a.conjugate()
b.makeCanonical()
b*a

a = MPS()
a.setRandomState()
b = 5*a + 7*a
b.makeCanonical()
print()
print(b.conjugate() * a)

a = MPS()
a.setRandomState()
b = 5*a - 7*a
b.makeCanonical()
print()
print(b.conjugate() * a)

c = MPO()
c.makeIdentity()

print(b.conjugate() * c * a)