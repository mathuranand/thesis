import numpy as np
from norms import normL2

def compute_fixed_point(T, v, error_tol=1e-4, maxiter=50, verbose=1, *args,
                        **kwargs):
	iterate = 0
	error = error_tol +1
	while iterate< maxiter and error> error_tol:
		new_v = T(v, *args, **kwargs)
		iterate +=1
		error = normL2(new_v,v, len(v))
		if verbose:
			print("Computed iterate %d with error %f" % (iterate, error))
		try:
			v[:] = new_v
		except TypeError:
			v = new_v
	return v
