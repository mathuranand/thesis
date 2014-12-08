import numpy as np
from scipy import integrate 
from scipy import interp
from norms import normL2

"""
	This class defines the primitives for the optimal spatial growth model. 
	The model is defined on the [0,1] interval. You can manipulate the 
	ker function to make the space periodic. 
	
	Parameters
	-----------
	f : 	function
			f: H --> R is the aggregate production function.  
	fp : 	function
			fp: H-->H. fp(x) for x\in H is the Frechet derivative at x
	Omega : 	function
			Omega: H-->H is a compact spillover operator 
			Omega(x)(i) tells me the spills received at i
	ker : 	function
			ker: RxR-->R kernal function defining my spillovers
	beta :	scalar(int),
			discount
	alpha :	scalar(int),
			boring
	delta :	scalar(int),
			even more boring
	theta :	scalar(int),
			CRRA parameter, who cares 
	grid_size : scalar(int), optional(default=10)
	M :		scalar(int),  optional(default =1)
			value of M in constraint \int_{0}_{1}x(i) =M, not so boring. 
			model results should not depend on value of M. 
	phi:	scalar(int), default = 1
			magnitude of decent direction in decent operator
						Tw = w- phi*fp(w)
			Starting at 1, at each iteration this 
			is updated using Armijo rule (2.2.2.1 in Ulbrich)
	
	Attributes
	----------
	f,fp,Omega,ker,beta,alpha,delta,theta,M : see Parameters
	grid : array_like(float, ndim=1)
			The grid over space.

"""

class geog(object):
	
	## you should input the f: H ->R function and the gradient of this function fp(x)\in H, so fp: H->H
	
	def __init__(self, f, fp,alpha, beta,delta, theta, Omega, ker, grid_size = 10, M = 1, phi=1, gamma = .5):
		
		self.f, self.fp, self.Omega, self.ker = f, fp, Omega, ker
		self.alpha, self.beta, self.delta, self.theta, self.M, self.phi, self.gamma = alpha, beta, delta, theta, M, phi, gamma
		self.grid = np.linspace(0,1, grid_size)
		
		
	def decent(self, w, phi):
	
		"""
		My decent operator. We want to do maximisation,
		so we min -f. The gradient is  -f', thus Tw = w+fp. 
		
		Parameters
		----------
		w : array_like(float, ndim=1)
		The value of the input function on different grid points
		fp,phi: see Class(?) 
		
		Issues
		--------
		Should Omega and ker be inputs for the decent function as *args?
		Note how phi is an input to this function. So self.phi is updated
		as per armijo rule and then entered into the decent function.  
		"""
		Tw = np.empty(len(w))
		gradient = self.fp(w, self.alpha,self.Omega,self.ker)
		for i in range(len(w)):
			
			Tw[i] = w[i] + phi*gradient[i]
		return Tw
		
	def proj(self, w, M):
		"""
		My projection operator. See Shanker(2014) x.x for first order 
		conditions 
		
		Parameters
		----------
		w : array_like(float, ndim=1)
		The value of the input function on different grid points
		fp,phi: see Class(?) 
		
		Issues
		-------
		Linear interpolation step may be questionable
		
		"""
		
		Aw = lambda x: interp(x, self.grid, w)
		z = integrate.quad(Aw,0,1)
		z = z[0]
		Pw = np.empty(len(w))
		for i in range(len(w)):
			Pw[i]=	self.M + w[i] - z
		return Pw
	
	def proj_dec(self,w):
		"""
		The projected decent operator is a composition of two operators.
		First is the decent
								Tw = w-phi*fp(w)
		second is projection 
								P(w) = argmin_x\inC |w-x|
		where C is our constraint set
		
		We choose phi according to the Armijo rule. 
		
		
		Parameters
		----------
		w : array_like(float, ndim=1)
		The value of the input function on different grid points
		
		Issues
		----------
		Why is not phi an input parameter for the function?What if it 
		was an input? Phi here is an attribute that belongs to an instance
		of geog. The proj_dec method will update phi. 
		"""	
		difference  = -1
		self.phi = 1
		while difference < 0: 
			print("Armijo coef is %f" % (self.phi))
			Tw = self.decent(w,self.phi)
			PTw = self.proj(Tw,self.M)
			difference = -self.f(PTw,self.alpha,self.Omega,self.ker) +self.f(w,self.alpha,self.Omega,self.ker) + (self.gamma/self.phi)*normL2(w,PTw,len(w))
			if self.phi<1:
				self.phi = (self.phi)**2
			else:
				self.phi = 1/2
		return PTw
	
	def growth(self, v_star):
		"""
		Because the opportunity cost of investment (interest rate) is 
		the same at each location, the frechet derivative will equal a 
		constant at optimal x. This value is the equilibrum 
				
				lambda = (1+g)\beta +\delta -1
		
		
		Parameters
		----------
		v_star : array_like(float, ndim=1)
		The value of the input function on different grid points
		
		"""	
		lamb   = self.fp(v_star,self.alpha, self.Omega, self.ker)
		g_star = ((lamb[0] +1 -self.delta)*self.beta)**(-self.theta)  -1
		return g_star, lamb
	
		



		
	
