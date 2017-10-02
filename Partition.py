# Xinran (Lisa) Liu
# the code for simulation in (Carlsson 2012)
# given a number of points in a polygon and an information map,
# partition the polygon into equitable parts, in which each subregion
# has same amount of information and exactly one point
import numpy as np
from scipy import stats
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
from PIL import Image
from tools import *

class partitioning(object):


	# image_name: string of the name of image
	# mu: ndarray = info distribution
	# partitions: int = number of partitions
	# dir: 1 or -1, indicating gradient ascent or gradient descent
	def __init__(self,image_name, mu, partitions, dir):
		self.image = np.asarray(Image.open(image_name))
		(yran,xran,n) = self.image.shape
		L = self.image.tolist()
		self.T = np.array(L)
		self.S = [(0,0),(0,yran),(xran,0),(xran,yran)]
		self.mu = mu
		delta = 1
		x = np.arange(0,xran,delta)
		y = np.arange(0,yran,delta)
		X, Y = np.meshgrid(x, y)
		self.G = (X,Y)
		self.partitions = partitions
		self.dir = dir

	# P: point list = point set
	# partition the terrain with given point list, only runs once
	def partition(self, P):
		# S: vertex list = graph/polygon
		# P: point list = point set
		# mu: ndarray = info distribution
		# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
		# T: ndarray = the terrain map
		# the main function for region partition
		# partition the polygon S into n equitable parts, with n as the 
		# number of points in P, so that each subregion contains the same
		# information as specified in mu and exactly one point, returns the 
		# sum of variance of all subregions
		def regionPartition(arg):
			(S,P,mu,G,T) = arg
			n = len(P)
			if n == 1:
				R = T[T[:,:,0] > 0]
				G = T[T[:,:,1] > 0]
				B = T[T[:,:,2] > 0]
				return [np.var(R) + np.var(G) + np.var(B)]
			if n % 2 == 0:
				res = part2(S, P, mu, G, n//2, T)
				if res == None: return []
				((s1,p1,mu1,T1),(s2,p2,mu2,T2)) = res
				var1 = regionPartition((s1, p1, mu1, G, T1))
				var2 = regionPartition((s2, p2, mu2, G, T2))
				return var1 + var2
			else:

				newp = gh(P) # newp is a point list
				p0 = newp[0]
				k = n//2
				# determine a geodesic G(x0,x1|S) through newp[0]
				# that has k = (n-1)/2 points lying strictly to
				# its right and left
				[x0, x1] = geodesic(p0, P, S)
				# try to partition into three subregions
				return part3(x0, x1, newp, P, G, S, mu, T)
		res = regionPartition((self.S,P,self.mu,self.G,self.T))
		return res

	# fun: a cost function that takes in an array like object and returns
	# a single value, examples include np.mean, np.var etc.
	# optimize the partition and save the figure
	def optimize(self, fun):
		# S: vertex list = graph/polygon
		# mu: ndarray = info distribution
		# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
		# T: ndarray = the terrain map
		# n: int = the number of points
		# dir: 1 or -1, indicating gradient ascent or descent
		# iteration: the number of set of points generated at each state
		# the function that finds the optimum partition given the terrain map
		# and information map
		def run(S, mu, G, T, n, dir, iteration = 10):
			# first assume the map is always a rectangle to start with
			(X, Y) = G
			# get the boundary
			(xmin, xmax) = (np.min(X), np.max(X))
			(ymin, ymax) = (np.min(Y), np.max(Y))
			xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
			xl, yl = (xmax - xmin), (ymax - ymin)
			# initialize a pool object so we can run the program in parallel
			p = Pool(iteration)
			minvar = -1
			# delta is the rate the step decreases at each iteration of gradient a/descent
			delta = 0.90
			convergence = (np.var(T[:,:,0]) + np.var(T[:,:,1]) + 
				np.var(T[:,:,2]))/4
			# the list that keeps track of all the vars
			varList = []
			for i in range(3):
				# randomly generate points to start with
				initialGuess = [[(rand(xmin,xmax),rand(ymin,ymax)) for i in range(n)]
				 for i in range(5)]
				initialVars = []
				# pick the minimum from some number of initial guesses
				for i in range(len(initialGuess)):
					v = fun(self.partition(initialGuess[i]))
					initialVars.append(v)
				P = initialGuess[initialVars.index(min(initialVars))][1]
				# take the mean of the variances, it can be changed to other cost functions
				var = fun(self.partition((S,P,mu,G,T)))
				if minvar == -1 or minvar > var:
					minvar = var
					minP = P.copy()
				changed = True
				rate = 0.01
				varList.append(var)
				oldvar = var
				count = 0
				lastP = P.copy()

				while(changed):
					# get the list of arguments for the different set of points
					args = [(S,newPoints(P,xmin,xmax,ymin,ymax),mu,G,T) for i in range(iteration)]
					# new set of points
					newPs = [arg[1] for arg in args]
					# run partition in parallel on all set of points in newPs
					varL = p.map(self.partition, args)
					varL = [fun(l) for l in varL]
					lastP = P.copy()
					for i in range(n):
						# get the coordinates of all of the points close to the original ones
						pl = [pt[i] for pt in newPs]
						xl = [x for (x,y) in pl]
						yl = [y for (x,y) in pl]
						(x,y) = P[i]
						xl.append(x)
						yl.append(y)
						varsL = varL + [var]
						# interpolate according to the points and their vars
						f = interpolate.interp2d(xl,yl,varsL)
						(partialx, partialy) = (partial(f,0,P[i]), partial(f,1,P[i]))
						# take a step
						newx, newy = x + rate * partialx * dir, y + rate * partialy * dir
						if (newx < xmin or newx > xmax or newy < ymin or
							newy > ymax): newx,newy = x,y
						P[i] = (newx, newy)
					var = fun(self.partition((S,P,mu,G,T)))
					# reset if var changes too much in the opposite direction
					if var - oldvar > - convergence * dir: 
						P = lastP[:]
						var = oldvar
					if (abs(var - oldvar)) < 5: count += 1
					else: count = 0
					# change step size if didn't move
					if var >= oldvar or oldvar - var > convergence: rate *= delta
					# some information, can be removed if not needed
					print("count",count,"var",var, "oldvar",oldvar)
					print("P",P,"lastP",lastP)
					# convergence criterion
					if count >= 6: changed = False
					if var < minvar:
						minvar = var
						minP = P.copy()
					oldvar = var
					varList.append(var)
			print(P)
			print(minP,minvar)
			print(varList)
			return minP

		minP = run(self.S, self.mu, self.G, self.T, self.partitions, self.dir)
		plt.clf()
		self.partition(minP)


np.set_printoptions(threshold=10e10)
# examples of how you would use the class
image = np.asarray(Image.open('Thoune.png'))
(yran,xran,n) = image.shape
delta = 1
x = np.arange(0,xran,delta)
y = np.arange(0,yran,delta)
X, Y = np.meshgrid(x, y)
# Z is the information function
Z1 = mlab.bivariate_normal(X, Y, 50, 50, 650, 410)
par = partitioning('Thoune.png',Z1,4,-1)

par.optimize(np.mean)


