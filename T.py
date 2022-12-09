#! /usr/bin/env python3

import matplotlib.pyplot as plt
from math import *
import numpy as np

# période houle ressentie à bord, houle de face
# g=9.81

# V_bat=np.linspace(0,25,26)
# T=np.linspace(3,20,1000)

# for v in V_bat :
#     T_bat=[t/(1+2*pi*v/(g*t)) for t in T]
#     plt.plot(T,T_bat)
# plt.show()

# print(V_bat[25])
# print(T_bat[0])

# sinus échantillonné
f=2
fe=100
n=int(fe/f)
print(n)
T=np.linspace(0,1/f,n+1)
sinus=[sin(2*pi*f*t) for t in T]

plt.plot(T,sinus)
plt.show()