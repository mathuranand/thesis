import numpy as np
from scipy import integrate 
from scipy import interp

def normL2(x,y, grid_size):
	grid = np.linspace(0,1, grid_size)
	pwdxy = (x - y)**2
	Apdxy = lambda x: interp(x, grid, pwdxy)
	norm = integrate.quad(Apdxy,0,1)
	norm = norm[0]**(1/2)
	return norm
