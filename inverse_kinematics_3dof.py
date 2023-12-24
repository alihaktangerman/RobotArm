import numpy as np

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
    
class roboarm3dof: #d0 == 0
    def __init__(self, a, d) -> None:
        self.a = a
        self.d = d
    def o0(self):
        return rotmatrix3(self.a[0], 2)
    def o1(self):
        return rotmatrix3(self.a[1], 1)@self.o0()
    def o2(self):
        return rotmatrix3(-self.a[2], 1)@self.o1()
    def p0(self):
        return np.array([.0,.0,.0])
    def p1(self):
        return self.p0() + self.d[0]*self.o0()[2,:]
    def p2(self):
        return self.p1() + self.d[1]*self.o1()[0,:]
    def p3(self):
        return self.p2() + self.d[2]*self.o2()[0,:]
    def go(self,x):
        x[2] -= self.d[0]
        r = np.linalg.norm(x)
        self.a[0] = np.arctan2(x[1],x[0])
        self.a[2] = -np.arccos((r**2 - self.d[1]**2 - self.d[2]**2)/(2*self.d[1]*self.d[2]))
        self.a[1] = np.arcsin(x[2]/r) + np.arctan2(self.d[2]*np.sin(self.a[2]),(self.d[1]+self.d[2]*np.cos(self.a[2])))
        return self

print(roboarm3dof(np.array([.0,np.pi/2,.0]),np.array([1.,1.,1.])).go(np.array([.2,.5,.7])).p3())
