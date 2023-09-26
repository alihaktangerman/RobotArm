import numpy as np
from math import factorial as fact
import pygame
from time import time, sleep
pi = 3.14159265359

class dualnum:
	def __init__(self, a, b):
		self.a = a
		self.b = b
	def __neg__(self):
		return dualnum(-self.a, -self.b)
	def __add__(self, other):
		return dualnum(self.a+other.a, self.b+other.b)
	def __sub__(self, other):
		return self+(-other)
	def __mul__(self, other):
		return dualnum(self.a*other.a, self.a*other.b + self.b*other.a)
	def __pow__(self, n):
		ret = dualnum(1,0)
		for i in range(n):
			ret = ret*self
		return ret
	def __truediv__(self, other):
		assert(other.b == 0)
		return dualnum(self.a/other.a, self.b/other.a)
	def __str__(self):
		return f'{self.a} + {self.b}*eps'
"""
class dualnummat:
    def __init__(self, arr):
        self.arr = arr
    def __matmul__(self, other):
        arr = [[dualnum(0,0) for i in range(sle)] for j in range(3)]
        for i in range(3):
            for j in range(3):
                arr[i][j] = sum(self.arr[i][k]*other.arr[k][j] for k in range(3))
        return dualnummat(arr)
    def __str__(self):
        maxlen = max(j for j in max(len(str(x)) for x in self.arr))
        ret = ''
        for i in range
"""
def f(n):
	return dualnum(fact(n),0)

def sin(x):
	if type(x) != dualnum:
		x = dualnum(x,1)
	if x.a > 2*pi:
		x.a %= 2*pi
	return x - x**3/f(3) + x**5/f(5) - x**7/f(7) + x**9/f(9) - x**11/f(11) + x**13/f(13) - x**15/f(15) + x**17/f(17) - x**19/f(19) + x**21/f(21)

def cos(x):
	if type(x) != dualnum:
		x = dualnum(x,1)
	if x.a > 2*pi:
		x.a %= 2*pi
	return dualnum(1,0) - x**2/f(2) + x**4/f(4) - x**6/f(6) + x**8/f(8) - x**10/f(10) + x**12/f(12) - x**14/f(14) + x**16/f(16) - x**18/f(18) + x**20/f(20) - x**22/f(22) 

def rotmatrix2(theta):
    return [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]

def rotmatrix3(theta, axis):
    r2 = rotmatrix2(theta)
    ret = [[dualnum(0,0) for _ in range(3)] for _ in range(3)]
    for i, ii in enumerate([(axis-1)%3,(axis+1)%3]):
        for j, jj in enumerate([(axis-1)%3,(axis+1)%3]):
            ret[ii][jj] = r2[i][j]
    ret[axis][axis] = dualnum(1,0)
    if axis==1:
        ret[0][2] = -ret[0][2]
        ret[2][0] = -ret[2][0]
    return np.array(ret)

def eyematrix3():
    return [[dualnum(1 if i==j else 0, 0) for i in range(3)] for j in range(3)]
"""
class affinepoin:
    def __init__(self, M, v):
        self.M = M
        self.v
    def __mul__(self):
        return affinepoin(self.v + self.M@other.v, self.M@other.M)
"""
"""
class kinmroboarm:
    def __init__(self, theta, armlen, basepos, conf):
        self.theta = theta 
        self.armlen = armlen
        self.basepos = basepos
        self.conf = conf
    def __posaffine(i, time):
        assert i > -1
        if i == 0:
            return affinepoin(self.basepos, rotmatrix3(self.theta[0](time),self.conf[0]))
        if i == self.n:
            return pos(i-1)*affinepoin(arm(self.arm[i], self.theta[i](time)), eyematrix3())
        return pos(i-1)*affinepoin(arm(self.arm[i], self.theta[i](time)), rotmatrix3(self.theta[i](time), self.conf[i]))
    def pos(i, time):
    	return self.__posaffine(i, time).v
"""

class roboarm6dof:
	def __init__(self, a, r, o):
		assert len(a) == 6
		self.a = a
		assert len(r) == 6
		self.r = [dualnum(x,0) for x in r]
		self.o = [dualnum(x,0) for x in o]
	def o0(self,t):
		return rotmatrix3(self.a[0](t), 2)
	def o1(self,t):
		#return rotmatrix3(self.a[1](t), 0)@self.o0(t)
		return np.tensordot(rotmatrix3(self.a[1](t),0), self.o0(t),axes=([0,1]))
	def o2(self,t):
		return np.tensordot(rotmatrix3(self.a[2](t),0), self.o1(t),axes=([0,1]))
	def o3(self,t):
		return np.tensordot(rotmatrix3(self.a[3](t),0), self.o2(t),axes=([0,1]))
	def o4(self,t):
		return np.tensordot(rotmatrix3(self.a[4](t),0), self.o3(t),axes=([0,1]))
	def o5(self,t):
		ret = np.tensordot(rotmatrix3(self.a[5](t),0), self.o4(t),axes=([0,1]))
		return ret
	def p0(self,t):
		return self.o
	def p1(self,t):
		return self.p0(t) + np.array([self.r[0]*x for x in self.o0(t)[2,:]])
	def p2(self,t):
		return self.p1(t) + np.array([self.r[1]*x for x in self.o1(t)[2,:]])
	def p3(self,t):
		return self.p2(t) + np.array([self.r[2]*x for x in self.o2(t)[2,:]])
	def p4(self,t):
		return self.p3(t) + np.array([self.r[3]*x for x in self.o3(t)[2,:]])
	def p5(self,t):
		return self.p4(t) + np.array([self.r[4]*x for x in self.o4(t)[2,:]])
	def p6(self,t):
		ret = self.p5(t) + np.array([self.r[5]*x for x in self.o5(t)[2,:]])
		#for x in ret: print(x, end='      ')
		#print('\n'*4)
		return ret

class projected_space:
    def __init__(self,rho,theta,d1,d2):
        self.rho = rho
        self.theta = theta
        self.d1 = d1
        self.d2 = d2
        self.viewer = np.array([0,0,0])
    def vec(self):
        ret = np.array([np.cos(self.rho)*np.cos(self.theta), np.cos(self.rho)*np.sin(self.theta), np.sin(self.rho)])
        return ret
    def __call__(self, p):
        vec = self.vec()
        viewer = (self.d2+self.d1)*vec
        self.viewer = viewer
        t = (self.d1-vec@viewer)/(vec@(p-viewer))
        loc =  viewer + t*p
        r_theta = np.array([-np.cos(self.rho)*np.sin(self.theta),np.cos(self.rho)*np.cos(self.theta),0.])
        r_theta /= np.linalg.norm(r_theta)
        r_rho = np.array([-np.sin(self.rho)*np.cos(self.theta),-np.sin(self.rho)*np.sin(self.theta),np.cos(self.rho)])
        r_theta /= np.linalg.norm(r_rho)
        A = np.array([r_theta,r_rho])
        P = np.linalg.inv(A@A.transpose())@A
        return P@loc
    def rot_up(self,s):
        self.rho += s*.001
    def rot_ho(self,s):
        self.theta += s*.001

class projectedroboarm6dof:
    def __init__(self, a, r, o, d1, d2):
        self.roboarm = roboarm6dof(a, r, o)
        self.ps = projected_space(.1,.7,d1,d2)
    def p0(self,t):
        ret = self.roboarm.p0(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def p1(self,t):
        ret = self.roboarm.p1(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def p2(self,t):
        ret = self.roboarm.p2(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def p3(self,t):
        ret = self.roboarm.p3(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def p4(self,t):
        ret = self.roboarm.p4(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def p5(self,t):
        ret = self.roboarm.p5(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def p6(self,t):
        ret = self.roboarm.p6(t)
        ret = self.ps(np.array([x.a for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v0(self,t):
        ret = self.roboarm.p0(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v1(self,t):
        ret = self.roboarm.p1(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v2(self,t):
        ret = self.roboarm.p2(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v3(self,t):
        ret = self.roboarm.p3(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v4(self,t):
        ret = self.roboarm.p4(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v5(self,t):
        ret = self.roboarm.p5(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    def v6(self,t):
        ret = self.roboarm.p6(t)
        ret = self.ps(np.array([x.b for x in ret]))
        return pygame.math.Vector2([x for x in ret])
    
y = 0
dir = 1
running = 1
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
linecolor = 255, 0, 0
vellineco = 0, 255, 0
bgcolor = 0, 0, 0

my_pra = projectedroboarm6dof([lambda x: x for i in range(6)],200*np.ones(6),np.array([20.,20.,20.]),-2000.,-500.)

t = time()
delta = .005

while time()-t < 10:
    tnow = time()
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        running = 0
    screen.fill(bgcolor)
    vec = pygame.math.Vector2(250,250)

    pygame.draw.line(screen, linecolor, vec+my_pra.p0(tnow-t), vec+my_pra.p1(tnow-t), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p1(tnow-t), vec+my_pra.p2(tnow-t), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p2(tnow-t), vec+my_pra.p3(tnow-t), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p3(tnow-t), vec+my_pra.p4(tnow-t), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p4(tnow-t), vec+my_pra.p5(tnow-t), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p5(tnow-t), vec+my_pra.p6(tnow-t), width=5)

    #pygame.draw.line(screen, linecolor, vec+my_pra.p0(tnow-t), vec+my_pra.p0(tnow-t)+my_pra.v0(t), width=5)
    #pygame.draw.line(screen, linecolor, vec+my_pra.p1(tnow-t), vec+my_pra.p1(tnow-t)+my_pra.v1(t), width=5)
    #pygame.draw.line(screen, linecolor, vec+my_pra.p2(tnow-t), vec+my_pra.p2(tnow-t)+my_pra.v2(t), width=5)
    #pygame.draw.line(screen, linecolor, vec+my_pra.p3(tnow-t), vec+my_pra.p3(tnow-t)+my_pra.v3(t), width=5)
    #pygame.draw.line(screen, linecolor, vec+my_pra.p4(tnow-t), vec+my_pra.p4(tnow-t)+my_pra.v4(t), width=5)
    pygame.draw.line(screen, vellineco, vec+my_pra.p6(tnow-t), vec+my_pra.p6(tnow-t)+my_pra.v6(tnow-t), width=2)

    pygame.display.flip()

