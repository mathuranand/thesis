from geog import geog
import numpy as np
from scipy import integrate,interp
import matplotlib.pyplot as plt

from compute_fp import compute_fixed_point

"""
filename: geoss_cdd.py
Author: Akshay Shanker 
"""


"""
	Implements the geo model with Cobb-Douglass production
"""

grid_size = 100
grid = np.linspace(0,1, grid_size)
beta = .9
theta = .5
delta = .1
alpha = .3
M=10


ker = lambda x,y: 1*np.exp(-(.2*(x-y)**2))

def Omega(x,ker):
	s = np.zeros(len(x))
	O = np.zeros(len(x))
	for i,k in enumerate(grid): 
		for j,l in enumerate(grid):
			##s[j] is the spillover from j to i##
			s[j] = ker(k,l)*x[j]
		As = lambda y: interp(y,grid,s)
		##O[i] is the sum of spillovers received by i##
		oi = integrate.quad(As,0,1)
		O[i] = float(oi[0])
	return O 
	
	
def f(x,alpha, Omega, *args):
	y = np.zeros(len(x))
	
	O = Omega(x,*args)
	
	for i in range(len(x)):
		y[i] = (x[i]**alpha)*O[i]**(1-alpha)
	
	Ay = lambda z: interp(z, grid, y)
	
	Y = integrate.quad(Ay,0,1)
	
	return Y[0]
	
def fp(x,alpha,Omega, *args): 
	y = np.zeros(len(x))
	S = Omega(x,*args)

	f2 = np.zeros(len(x))
	f_2 = np.zeros(len(x))
	
	for i in range(len(x)):
		f_2[i] = (1-alpha)*(x[i]**alpha)*S[i]**(-alpha)
	f2 = Omega(f_2,*args)
	for i in range(len(x)):
		y[i] = (alpha*x[i]**(alpha-1))*S[i]**(1-alpha) + f2[i]
	return y


x = np.ones(grid_size)
x = x*M

geo = geog(f,fp,alpha, beta, delta, theta, Omega, ker, grid_size,M)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlabel("location",fontsize = 14)
ax.set_ylabel("capital",fontsize = 12)

v_star = compute_fixed_point(geo.proj_dec, x, maxiter=1000)
g_star, lamb = geo.growth(v_star)

ax.plot(grid,v_star, '-', lw=1)

plt.savefig('geo.eps')
