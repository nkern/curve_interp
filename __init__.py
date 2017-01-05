"""
Localized (Nearest Neighbor) Polynomial Interpolation of a smooth function
Nicholas Kern
2016
"""

import numpy as np
import numpy.linalg as la
import itertools
import operator
import functools

def poly_design_mat(Xrange,dim=2,degree=6):
	"""
	- Create polynomial design matrix given discrete values for dependent variables
	- dim : number of dependent variables 
	- degree : degree of polynomial to fit
	- Xrange is a list with dim # of arrays, with each array containing
		discrete values of the dependent variables that have been unraveled for dim > 1
	- Xrange has shape dim x Ndata, where Ndata is the # of discrete data points
	- A : Ndata x M design matrix, where M = (dim+degree)!/(dim! * degree!)
	- Example of A for dim = 2, degree = 2, Ndata = 3:
		A = [   [ 1  x  y  x^2  y^2  xy ]
				[ 1  x  y  x^2  y^2  xy ]
				[ 1  x  y  x^2  y^2  xy ]   ]
	"""

	# Generate all permutations
	perms = itertools.product(range(degree+1),repeat=dim)
	perms = np.array(map(list,perms))

	# Take the sum of the powers, sort, and eliminate sums > degree
	sums = np.array(map(lambda x: reduce(operator.add,x),perms))
	argsort = np.argsort(sums)
	sums = sums[argsort]
	keep = np.where(sums <= degree)[0]
	perms = perms[argsort][keep]

	# Create design matrix
	to_the_power = lambda x,y: np.array(map(lambda z: x**z,y))
	dims = []
	for i in range(dim):
		dims.append(to_the_power(Xrange[i],perms.T[i]).T)
	dims = np.array(dims)

	A = np.array(map(lambda y: map(lambda x: functools.reduce(operator.mul,x),y),dims.T)).T

	return A

def chi_square_min(y,A,N):
	'''
	- perform chi square minimization
	- A is data model
	- N are weights of each y_i for fit
	- y are dataset
	'''
	# Solve for coefficients xhat
	xhat = np.dot( la.inv( np.dot( np.dot(A.T,la.inv(N)), A)), np.dot( np.dot(A.T,la.inv(N)), y) )

	return xhat

def get_nearest(x,xarr,x_id,y_curve,n=3):
	"""
	Get n nearest points in xarr to point x, and return their IDs in increasing order
	so long as y_val != nan *for all curves*
	"""
	not_nan = np.array(map(lambda x: True not in np.isnan(x), y_curve))
	xarr = xarr[not_nan]
	x_id = x_id[not_nan]
	dist = np.abs(xarr-x)
	nn_id = x_id[np.argsort(dist)][:n]
	return nn_id[np.argsort(nn_id)]

def curve_interp(x_array, x_curve, y_curve, n=3, degree=2, extrap_deg=1, extrap_n=2):
	""" 
	curve_interp(x_array, x_curve, y_curve, n=3, degree=2, extrap_deg=1, extrap_n=2)
	- Interpolate smooth curve(s) via localized polynomial regression
	- Fit a polynomial of <degree> degree to <n> nearest points
	x_array : row vector (ndarray) of desired x points at which we interpolate the curve
	x_curve : row vector (ndarray) of x-values of the curve we wish to interpolate, with length x_num
	y_curve : matrix (ndarray) with shape (x_num, c_num), containing y-values of curve(s) we wish to interpolate

	n : number of points to use in fit
	degree : degree of polynomial fit
	extrap_deg : degree of polynomial fit when extrapolating
	extrap_n : number of points to use in extrapolating fit

	- Note there can be multiple curves we independently fit for 
		simultaneously--c_num is the number of curves we fit for--but their y-values 
		must all be evaluated at the *same* x-values.
	"""

	# Order data by xvalues
	sort = np.argsort(x_array)
	x_array = x_array[sort]
	sort = np.argsort(x_curve)
	x_curve = x_curve[sort]
	y_curve = y_curve[sort]

	# Get numbers of x-values and curves
	x_num = len(x_curve)
	try: c_num = y_curve.shape[1]
	except IndexError: c_num = 1

	# Assign each x_curve point an identification number
	x_id = np.arange(x_num)

	# Iterate over desired points to interpolate
	y_interp = []
	# Set interpolation by default
	interpolating = True
	for i in range(len(x_array)):

		# Fit flag
		fit = True

		# Assign x point
		x = x_array[i]

		if interpolating == True:
			n_fit = n
			poly_deg = degree
		else:
			n_fit = extrap_n
			poly_deg = extrap_deg

		# Get nearest neighbors
		if i != 0:
			# If nearest neighbors haven't changed, do redo fit! If poly_deg has changed, re-do fit!
			nn_id_new = get_nearest(x,x_curve,x_id,y_curve,n=n_fit)
			if np.abs(sum(nn_id - nn_id_new)) < 0.0001:
				fit = False
			else:
				# If they have, get new nearest neighbors
				nn_id = nn_id_new
		else:
			nn_id = get_nearest(x,x_curve,x_id,y_curve,n=n_fit)

		# Check for interpolation or extrapolation
		x_diff = x_array[i] - x_curve[nn_id]
		sum1 = np.abs(np.sum(x_diff))
		sum2 = np.sum(np.abs(x_diff))
		if np.abs(sum1 - sum2) < 0.0001:
			interpolate = False
			if interpolating == True:	redo_fit = True
			else:						redo_fit = False
			interpolating = False
		else:
			interpolate = True
			if interpolating == True:	redo_fit = False
			else:						redo_fit = True
			interpolating = True

		if interpolating == True:
			n_fit = n
			poly_deg = degree
		else:
			n_fit = extrap_n
			poly_deg = extrap_deg

		if redo_fit == True:
			nn_id = get_nearest(x,x_curve,x_id,y_curve,n=n_fit)
			fit = True

		if fit == True:
			A = poly_design_mat([x_curve[nn_id]],dim=1,degree=poly_deg)
			N = np.eye(n_fit)
			xhat = chi_square_min(y_curve[nn_id],A,N)

		# Make prediction
		A = poly_design_mat([[x]],dim=1,degree=poly_deg)
		y_pred = np.dot(A,xhat)

		y_interp.append(y_pred.ravel())

	return np.array(y_interp)

