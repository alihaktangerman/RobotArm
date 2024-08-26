import numpy as np
import pygame
from time import time, sleep
import random

def rotmatrix2(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def rotmatrix3(theta, axis):
    r2 = rotmatrix2(theta)
    ret = np.zeros([3,3])
    for i, ii in enumerate([(axis-1)%3,(axis+1)%3]):
        for j, jj in enumerate([(axis-1)%3,(axis+1)%3]):
            ret[ii][jj] = r2[i][j]
    ret[axis][axis] = 1
    if axis==1:
        return ret.transpose()
    return ret

class roboarm6dof:
    def __init__(self, a, r, o):
        assert len(a) == 6
        self.a = np.array(a)
        assert len(r) == 6
        self.r = np.array(r)
        self.o = np.array(o)
    def o0(self):
        return rotmatrix3(self.a[0], 2)
    def o1(self):
        return rotmatrix3(self.a[1], 0)@self.o0()
    def o2(self):
        return rotmatrix3(self.a[2], 2)@self.o1()
    def o3(self):
        return rotmatrix3(self.a[3], 0)@self.o2()
    def o4(self):
        return rotmatrix3(self.a[4], 2)@self.o3()
    def o5(self):
        return rotmatrix3(self.a[5], 0)@self.o4()
    def p0(self):
        return self.o
    def p1(self):
        return self.p0() + self.r[0]*self.o0()[2,:]
    def p2(self):
        return self.p1() + self.r[1]*self.o1()[2,:]
    def p3(self):
        return self.p2() + self.r[2]*self.o2()[2,:]
    def p4(self):
        return self.p3() + self.r[3]*self.o3()[2,:]
    def p5(self):
        return self.p4() + self.r[4]*self.o4()[2,:]
    def p6(self):
        return self.p5() + self.r[5]*self.o5()[2,:]
        
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
        r_rho /= np.linalg.norm(r_rho)
        A = np.array([r_theta,r_rho])
        P =  np.linalg.inv(A@A.transpose())@A
        return P@loc
    def rot_up(self,s):
        self.rho += s*.001
    def rot_ho(self,s):
        self.theta += s*.001

class projectedroboarm6dof:
    def __init__(self, a, r, o, d1, d2):
        self.roboarm = roboarm6dof(a, r, o)
        self.ps = projected_space(.1,.7,d1,d2)
    def p0(self):
        ret = self.roboarm.p0()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    def p1(self):
        ret = self.roboarm.p1()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    def p2(self):
        ret = self.roboarm.p2()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    def p3(self):
        ret = self.roboarm.p3()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    def p4(self):
        ret = self.roboarm.p4()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    def p5(self):
        ret = self.roboarm.p5()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    def p6(self):
        ret = self.roboarm.p6()
        ret = [_ for _ in self.ps(ret)]
        return pygame.math.Vector2(ret)
    
y = 0
dir = 1
running = 1
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
linecolor = 255, 0, 0
bgcolor = 0, 0, 0

my_pra = projectedroboarm6dof(np.array([1.0,-1.0,0.0,1.0,0.0,1]),200*np.ones(6),np.array([20.,20.,20.]),-2000.,-500.)

t = time()
delta = .005

while time()-t < 60:
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        running = 0
    screen.fill(bgcolor)
    vec = pygame.math.Vector2(250,250)
    pygame.draw.line(screen, linecolor, vec+my_pra.p0(), vec+my_pra.p1(), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p1(), vec+my_pra.p2(), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p2(), vec+my_pra.p3(), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p3(), vec+my_pra.p4(), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p4(), vec+my_pra.p5(), width=5)
    pygame.draw.line(screen, linecolor, vec+my_pra.p5(), vec+my_pra.p6(), width=5)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        my_pra.ps.rot_up(3)
    if keys[pygame.K_DOWN]:
        my_pra.ps.rot_up(-3)
    if keys[pygame.K_RIGHT]:
        my_pra.ps.rot_ho(3)
    if keys[pygame.K_LEFT]:
        my_pra.ps.rot_ho(-3)
    
    if keys[pygame.K_q]:
        my_pra.roboarm.a[0] += .003
    if keys[pygame.K_w]:
        my_pra.roboarm.a[1] += .003
    if keys[pygame.K_e]:
        my_pra.roboarm.a[2] += .003
    if keys[pygame.K_r]:
        my_pra.roboarm.a[3] += .003
    if keys[pygame.K_t]:
        my_pra.roboarm.a[4] += .003
    if keys[pygame.K_y]:
        my_pra.roboarm.a[5] += .003    

    if keys[pygame.K_a]:
        my_pra.roboarm.a[0] -= .003
    if keys[pygame.K_s]:
        my_pra.roboarm.a[1] -= .003
    if keys[pygame.K_d]:
        my_pra.roboarm.a[2] -= .003
    if keys[pygame.K_f]:
        my_pra.roboarm.a[3] -= .003
    if keys[pygame.K_g]:
        my_pra.roboarm.a[4] -= .003
    if keys[pygame.K_h]:
        my_pra.roboarm.a[5] -= .003

    pygame.display.flip()

