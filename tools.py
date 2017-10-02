import numpy as np
from scipy import stats
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
from PIL import Image

# check if two numbers are almost equal to each other
def almostEq(a,b, epsilon = 1e-4):
	if abs(a-b) <= epsilon: return True
	return False

# the following two functions are cited from
# https://stackoverflow.com/questions/2827393/
# angles-between-two-n-dimensional-vectors-in-python
# return the unit vector of the vector
def unit_vector(vector):
	return vector / np.linalg.norm(vector)

# find the angle between two vectors
def angle_between(v1, v2):
	if (v1 == (0,0) or v2 == (0,0)): return 0
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# S: vertex list = graph/polygon
# find the interior(average) of vertices
def findInterior(S):
	(x,y) = (sum(x for (x,y) in S), sum(y for (x,y) in S))
	return (x/len(S), y/len(S))


# key of HOF sort function
# get the angle between (1,0) and the vector from interior to v
def sorthelp(interior,v):
	(a,b),(x,y) = interior,v
	if y - b < 0: return 2 * np.pi - angle_between((1,0), (x-a,y-b))
	if y == b and x - a < 0: return np.pi
	return angle_between((1,0), (x-a,y-b))


# S: vertex list = graph/polygon
# sort vertices of a convex polygon
def vsort(S, v = None):
	if v == None:
		(x,y) = findInterior(S)
	else:
		(x,y) = v
	S = sorted(S, key = lambda v: sorthelp((x,y),v))
	temp = [angle_between((1,0),(a-x,b-y)) for (a,b) in S]
	return S


# u: coord tuple = x,y coordinate
# v: coord tuple = x,y coordinate
# P: point list = point set
# count points in the left shell of u,v
def countPtsLeft(u, v, P):
	res = []
	((x1, y1), (x2, y2)) = (u,v)
	# when x coordinates are equal
	if x1 == x2:
		for (x,y) in P:
			# compare y coordinates and see which part
			# to count
			if x <= x1 and y1 <= y2: res.append((x,y))
			elif x >= x1 and y1 >= y2: res.append((x,y))
	else:
		# x coordinates are different, check direction
		# to see if the point lies above or below the line
		coeffs = np.polyfit((x1,x2),(y1,y2),1)
		poly = np.poly1d(coeffs)
		if (x1 < x2):
			for (x,y) in P:
				if y > poly(x) or almostEq(y, poly(x)):
					res.append((x,y))
		else:
			for (x,y) in P:
				if y < poly(x) or almostEq(y, poly(x)):
					res.append((x,y))
	return (len(res),res)



# G: (X, Y) 2-d array tuple = coordinate of x and y axis, meshgrid
# mu: ndarray = info distribution
# u,v: coord tuple = x,y coordinate
# find out the area in the left shell of G(u,v|S)
# basically doing the same thing as count points except taking
# the sum of the information
def lArea(G, mu, u, v):
	((x1, y1), (x2, y2)) = (u,v)
	(X,Y) = G
	if x1 == x2:
		if (y1 <= y2):
			left = X <= x1
		else:
			left = X >= x1
	else:
		coeffs = np.polyfit((x1,x2),(y1,y2),1)
		poly = np.poly1d(coeffs)
		if (x1 < x2):
			left = np.logical_or(Y > poly(X), abs(Y-poly(X)) < 1e-4)
		else:
			left = np.logical_or(Y < poly(X), abs(Y-poly(X)) < 1e-4)
	return np.sum(mu[left])



# u: coord tuple = x,y coordinate
# a: num = area/info, < total info
# S: vertex list = graph/polygon
# mu: ndarray = info distribution
# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
def LShell(u, a, S, mu, G):
	(x, y) = u
	# sort vertices of the polygon
	T = vsort(S)
	# get area on the left of every vertex
	S = sorted(S, key = lambda v: lArea(G, mu, u, v))
	L = [lArea(G, mu, u, v) for v in S]
	assert(L == sorted(L))
	lo, hi, mid = 0, len(S) - 1, 0
	# look for the index of vertex that's just smaller than a 
	# using binary search
	while lo < hi:
		mid = (lo + hi) // 2
		if L[mid] <= a and (mid + 1 < len(S) and L[mid+1] >= a):
			break
		else:
			if L[mid] > a: hi = mid
			else: lo = mid + 1
	assert(mid + 1 == len(S) or (L[mid] <= a and L[mid+1]>=a))
	if mid + 1 == len(S): print("Can't happen, this is 1-d")
	else:
		# look for the exact point using binary search again
		lo, hi = S[mid], S[mid + 1]
		v = ((lo[0] + hi[0])/2,(lo[1]+hi[1])/2)
		area = lArea(G, mu, u, v)
		while(not (almostEq(lo[0], hi[0], 0.1) and almostEq(lo[1], hi[1], 0.1))):
			v = ((lo[0] + hi[0])/2,(lo[1]+hi[1])/2)
			area = lArea(G,mu,u,v)
			if almostEq(area,a,0.1): break
			if area < a:
				lo = v
			else:
				hi = v
	return v



# x0, x1: points that constitute of the segment that partitions the polygon
# mu: ndarray = info distribution
# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
# returns two info distribution that corresponds to the info distribution of
# the left and right parts
def partitionInfo(x0, x1, mu, G):
	(X,Y) = G
	((a,b),(c,d)) = (x0, x1)
	dim = len(mu.shape)
	# get the info distribution for both polygons
	if a == c:
		if (b <= d):
			left = X <= a
		else:
			left = X >= a
	else:
		# the polynomial for (x0,x1)
		coeffs = np.polyfit([a,c], [b,d], 1)
		poly = np.poly1d(coeffs)
		if (a < c):
			left = np.logical_or(Y > poly(X), abs(Y-poly(X)) < 1e-4)
		else:
			left = np.logical_or(Y < poly(X), abs(Y-poly(X)) < 1e-4)
	# zero out entries outside polygon
	mu2 = mu.copy()
	right = left != True
	mu1 = mu.copy()
	if dim == 3:
		mu2[left] = -1
		mu1[right] = -1		
	else:
		mu2[left] = 0
		mu1[right] = 0
	return mu1, mu2



# x0, x1, points on the boundary such that the line in between them 
# partitions the polygon
# S: vertex list = graph/polygon
# P: point list = point set
# mu: ndarray = info distribution
# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
# T: ndarray = the terrain map
# returns the tuple ((s1,p1,mu1,T1),(s2,p2,mu2,T2)) in which s1,s2 are 
# the vertices of the two new polygons, p1,p2 are the point set in the 
# new polygons, and mu1, mu2 are the normalized info distribution of the 
# new polygons, it also draws a line to show the partition
# the actual partition function
def partition(x0, x1, mu, S, G, P, T):
	# get points and vertices on left and right of (x0,x1)
	(nl, p1) = countPtsLeft(x0,x1,P)
	(nr, p2) = countPtsLeft(x1,x0,P)
	(vl, s1) = countPtsLeft(x0,x1,S)
	(vr, s2) = countPtsLeft(x1,x0,S)
	# x0,x1 will be new vertices of the polygons
	s1 += [x0, x1]
	s2 += [x0, x1]
	# remove duplicates
	s1 = list(set(s1))
	s2 = list(set(s2))
	(X,Y) = G
	((a,b),(c,d)) = (x0, x1)	

	# get the info distribution for both polygons
	mu1, mu2 = partitionInfo(x0, x1, mu, G)
	# normalize the distribution
	mu1 = mu1/np.sum(mu1)
	mu2 = mu2/np.sum(mu2)
	# get the terrain for both polygons
	T1, T2 = partitionInfo(x0, x1, T, G)
	# draw the line to partition
	x_axis = np.ogrid[a:c:200j]
	if (a != c):
		coeffs = np.polyfit([a,c], [b,d], 1)
		poly = np.poly1d(coeffs)
		y_axis = poly(x_axis)
	else:
		y_axis = np.ogrid[b:d:200j]
	plt.plot(x_axis,y_axis)
	return ((s1,p1,mu1,T1),(s2,p2,mu2,T2))

# S: vertex list = graph/polygon
# P: point list = point set
# mu: ndarray = info distribution
# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
# T: ndarray = the terrain map
# returns the tuple ((s1,p1,mu1,T1),(s2,p2,mu2,T2)) in which s1,s2 are 
# the vertices of the two new polygons, p1,p2 are the point set 
# in the new polygons, and mu1, mu2 are the normalized info distribution 
# of the new polygons, it also draws a line to show the partition
def part2(S, P, mu, G, k, T):
	n = len(P)
	if n == 0: return 
	# "arbitrarily" choose one point on boundary
	(x,y) = findInterior(S)
	# sort S and find a point on boundary which has the same x coordinate
	S = vsort(S)
	poly = np.poly1d([0,0])
	for i in range(len(S)):
		if i == len(S) - 1:
			j = 0
		else: j = i + 1
		first, second = S[i], S[j]
		# look for a boundary we can project x on
		if (min(first[0],second[0]) <= x <= max(first[0], second[0])):
			coeffs = np.polyfit([S[i][0],S[j][0]], [S[i][1],S[j][1]], 1)
			poly = np.poly1d(coeffs)
			break
	# initialize x0 and x1
	x0 = (x, poly(x))
	x1 = LShell(x0, k/n, S, mu, G)
	(pts, p) = countPtsLeft(x0, x1, P)
	# if equal, just partition it
	if pts == k:
		return partition(x0, x1, mu, S, G, P, T)
	found = False
	# traverse all of the boundaries by traversing all of the vertices and
	# check the segment between the two vertices
	for i in range(0, len(S)):
		if i == len(S) - 1: l = 0
		else: l = i + 1
		x_axis = np.ogrid[S[i][0]:S[l][0]:300j]
		if S[i][0] == S[l][0]: 
			y_axis = np.ogrid[S[i][1]:S[l][1]:300j]
		else:
			coeffs = np.polyfit((S[i][0],S[l][0]), (S[i][1],S[l][1]), 1)
			poly = np.poly1d(coeffs)
			y_axis = poly(x_axis)
		# traverse the boundary to see if we can find an equitable partition
		# brute force
		for j in range(1,299):
			u = (x_axis[j],y_axis[j])
			v = LShell(u,k/n,S,mu,G)
			if countPtsLeft(u,v,P)[0] == k:
				found = True
				break
		if found:
			break
	if not found:
		print("can't find a partition in this case")
	return partition(u,v,mu,S,G,P,T)



# p1, p2, p3: points that consitutes the turn
# return > 0 if left turn, < 0 if right turn, = 0 if collinear
def turn(p1, p2, p3):
	(x1,y1),(x2,y2),(x3,y3) = p1, p2, p3
	return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)



# P: point list = point set
# looks for a convex geodesic hull of P
def gh(P):
	# look for the point with smallest y coordinate
	# if y coordinates are the same, 
	# look for the point with smalllest x coordinate
	start = P[0]
	for (x,y) in P:
		if y < start[1]:
			start = (x,y)
		elif y == start[1] and x < start[0]:
			start = (x,y)
	# sort P according to the angle between the points and the
	# start make with x axis
	P = vsort(P, start)
	res = [start]
	for v in P:
		# avoid duplicate
		if start == v: continue
		# just add vertices if the algorithm just starts
		if len(res) <= 1: 
			res.append(v)
			continue
		res.append(v)
		assert(len(res) >= 3)
		# check if the last three vertices constitute a right turn
		# if so, remove the second to last vertex
		if turn(res[-3], res[-2], res[-1]) < 0:
			res.pop(-2)
	return res




# u, v: vertices that need to be fit on boundary
# S: vertex list = graph/polygon
# given two points, extend them to boundary and return the points on boundary
def boundaryFit(u, v, S):
	res = []
	# if u and v have same x coordinates
	if (u[0] == v[0]):
		for i in range(len(S)):
			if len(res) == 2: break
			if i == len(S) - 1:
				j = 0
			else: j = i + 1
			a,b = S[i], S[j]
			# if two end points have same x coordinates
			if (a[0] == b[0]):
				if a[0] != u[0]: continue
				else: return [a,b]
			if min(a[0], b[0]) - 0.1 <= u[0] <= max(a[0], b[0]) + 0.1:
				coeffs = np.polyfit([a[0],b[0]],[a[1],b[1]],1)
				polyS = np.poly1d(coeffs)
				res.append((u[0], polyS(u[0])))
		# there must be two points on boundary that they intersect
		assert(len(res) == 2)
		return res
	# u and v have different x coordinates
	coeffs = np.polyfit([u[0],v[0]],[u[1],v[1]],1)
	poly = np.poly1d(coeffs)
	S = vsort(S)
	for i in range(len(S)):
		if len(res) == 2: break
		if i == len(S) - 1:
			j = 0
		else: j = i + 1
		a,b = S[i], S[j]
		# if two end points have same x coordinates
		if a[0] == b[0]:
			if min(a[1],b[1]) - 0.1 <= poly(a[0]) <= max(a[1],b[1]) + 0.1:
				res.append((a[0], poly(a[0])))
			continue
		coeffs = np.polyfit([S[i][0], S[j][0]],[S[i][1], S[j][1]],1)
		polyS = np.poly1d(coeffs)
		x = np.roots(polyS - poly)
		if min(S[i][0], S[j][0]) - 0.1 <= x <= max(S[i][0], S[j][0]) + 0.1:
			res.append((x[0], poly(x[0])))
	if len(res) != 2:
		print("can't find a boundary fit", u,v)
	return res




# p0: the point through which the resulting line has half points on
# both side
# S: vertex list = graph/polygon
# P: point list = point set
# returns (x0,x1) such that (x0,x1) passes through p0 and there are
# half points in P on both side (excluding p0 itself)
def geodesic(p0, P, S):
	P = vsort(P, p0)
	k = len(P) // 2
	P.remove(p0)
	v = ((P[k-1][0] + P[k][0])/2, (P[k-1][1] + P[k][1])/2)
	return boundaryFit(p0, v, S)





# S: vertex list = graph/polygon
# P: point list = point set
# mu: ndarray = info distribution
# G: (X, Y) 1-d array tuple = coordinate of x and y axis, meshgrid
# ghp: the geodesic hull of P
# x0, x1: two points that on both side of the segment there are exactly
# half number of points
# T: ndarray = the terrain map
# partition the polygon into three equitable parts
def part3(x0, x1, ghp, P, G, S, mu, T):
	n = len(P)
	k = n//2
	# extend p0, p1 and p0, pn
	p0, p1, pn = ghp[0], ghp[1], ghp[-1]
	[u1, v1] = boundaryFit(p0, p1, S)
	if countPtsLeft(u1,v1,P)[0] > 2:
		u1, v1 = v1, u1
	[u2, v2] = boundaryFit(p0, pn, S)
	if countPtsLeft(u2,v2,P)[0] > 2:
		u2, v2 = v2, u2
	if countPtsLeft(u1, v1, [x0])[0] != 1:
		x0, x1 = x1, x0	

	# partition into three parts
	U1, S1 = partitionInfo(u1, v1, mu, G)
	L1, R1 = partitionInfo(x1, x0, S1, G)
	U2, S2 = partitionInfo(u2, v2, mu, G)
	L2, R2 = partitionInfo(x1, x0, S2, G)
	au1, al1, ar1 = np.sum(U1), np.sum(L1), np.sum(R1)
	au2, al2, ar2 = np.sum(U2), np.sum(L2), np.sum(R2)

	xL, xR = partitionInfo(x1, x0, mu, G)
	axL, axR = np.sum(xL), np.sum(xR)
	# check if thm4 can apply
	if (axL < k/n or axR < k/n):
		# partition the polygon into two parts, one of which has
		# only one point in there
		if part2 == None: return []
		((s1,p1,mu1,T1),(s2,p2,mu2,T2)) = part2(S, P, mu, G, 1, T)
		var1 = regionPartition((s1, p1, mu1, G, T1))
		var2 = regionPartition((s2, p2, mu2, G, T2))
		return var1 + var2

	# check if thm4 can apply
	# if so, apply thm4
	if not (au1 > 1/n and au2 > 1/n):
		# partition the polygon into two parts, one of which has
		# only one point in there
		res = part2(S,P,mu,G,1,T)
		if res == None: return []
		((s1,p1,mu1,T1),(s2,p2,mu2,T2)) = res
		var1 = regionPartition((s1, p1, mu1, G, T1))
		var2 = regionPartition((s2, p2, mu2, G, T2))
		return var1 + var2
	# partition the vertices in S into two parts with respect to 
	# segment x1, x0
	(snl, sl) = countPtsLeft(x1, x0, S)
	(snr, sr) = countPtsLeft(x0, x1, S)
	sl.extend([x1,x0])
	sr.extend([x0,x1])
	sl, sr = list(set(sl)), list(set(sr))
	# find the point in both subregions partitioned by x0, x1
	# so that they form an equitable partition
	newxl = LShell(p0, k/n, sl, xL, G)
	# partition for right part
	ar = 1/n - (axL - k/n)
	newxr = LShell(p0, ar, sr, xR, G)
	(nl,Ptemp) = countPtsLeft(newxl,p0,P)
	(nl,Up) = countPtsLeft(p0,newxr,Ptemp)
	if nl != 1:
		((s1,p1,mu1,T1),(s2,p2,mu2,T2)) = part2(S, P, mu, G, 1, T)
		var1 = regionPartition((s1, p1, mu1, G, T1))
		var2 = regionPartition((s2, p2, mu2, G, T2))
		return var1 + var2
	((sl1, pl, mu1, T1),(sr1,pr,mur1, T2)) = partition(x1, p0, mu, S,
		G, P, T)
	((sr2,pr2,mur2,Tr2), (sr1, pr1, mur1,T2)) = \
	partition(p0, newxr, xR, sr, G, pr, T2)

	((sl1,pl1,mul1,T1),(sl2, pl2, mul2, Tl2)) = \
	partition(p0, newxl, xL, sl, G, pl, T1)
	pl1.remove(p0)

	pr1.remove(p0)
	newT = Tl2 + Tr2
	temp = newT != 0
	var3 = np.var(newT[temp])
	# recursion
	var1 = regionPartition((sl1, pl1, mul1, G, T1))
	var2 = regionPartition((sr1, pr1, mur1, G, T2))
	return var1 + var2 + var3


# mini, maxi: int = the range of random numbers we want to generate
# generate random numbers with in the range of (mini,maxi)
def rand(mini, maxi):
	c = (mini + maxi) / 2
	l = (maxi - mini) / 2
	return (random.random() - 0.5) * l + c


# P: point list = the (x,y) coordinate list of points
# epsilon: int = the new points will be with in the range of +-epsilon
# randomly generate new point list so that each point is with in the range
# of epsilon with the old point
def newPoints(P, xmin, xmax, ymin, ymax, epsilon = 0.5):
	res = []
	for (x,y) in P:
		(newx, newy) = (rand(x - epsilon, x + epsilon),
			rand(y - epsilon, y + epsilon)) 
		while not(xmin < newx < xmax and ymin < newy < ymax):
			(newx, newy) = (rand(x - epsilon, x + epsilon), 
				rand(y - epsilon, y + epsilon))
		res.append((newx,newy))
	return res

# f: the function that takes in two inputs (x,y)
# var: the variable we are taking the partial derivative on
# coord: (x,y) coordinate we are taking the partial derivative on
# delta: the step size
def partial(f, var, coord):
	f = np.vectorize(f)
	(x,y) = coord
	a = np.arange(x - 0.5, x + 0.5, 0.1)
	b = np.arange(y - 0.5, y + 0.5, 0.1)
	(X,Y) = np.meshgrid(a,b)
	res = f(X,Y)
	grad = np.gradient(res)
	[Yg,Xg] = grad
	if var == 0: return Xg[5][5]
	else: return Yg[5][5]
