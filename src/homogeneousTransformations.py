import numpy as np


def skew(v):
    return np.float32([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])


class HTransf(object):
    """
            Class for representing and manipulating homogeneous transformation matrices in SE(3)
            constructor args: (a) angle, (k) axis, (t) translation,
            alternativly use the from_matrix() or from_vecs() methods to create a class object from an existing ndarray represeinting
            a homogeneous transformation matrix in SE(3) 
    """

    def __init__(self, a=0, k=np.array([1, 1, 1]), t=np.array([0, 0, 0])):
        self.angle = a
        self.axis = (k/np.linalg.norm(k)).reshape(3, 1)
        self.skew_axis = skew(self.axis)
        self.t = t
        self.rot_matrix = np.identity(
            3) + np.sin(a)*self.skew_axis + (1-np.cos(a))*(self.skew_axis@self.skew_axis)
        self.matrix = np.zeros((4, 4))
        self.matrix[0:3, 0:3] = self.rot_matrix
        self.matrix[0:3, 3] = t.reshape(3)
        self.matrix[3, :] = np.array([0, 0, 0, 1])

    @classmethod
    def from_matrix(cls, m):
        '''allows construction from  honogeneous transformation matrix'''
        if m.shape == (3, 3):
            t = np.array([0, 0, 0])
        else:
            t = np.array([m[0][3], m[1][3], m[2][3]])
        trR = m[0][0]+m[1][1]+m[2][2]
        if(trR > 3.0):
            trR = 3.0
        if(trR < -1):
            trR = -1.0
        if(np.isclose(trR, -1.0, rtol=1e-4)):  # trR==-1):
            a = np.pi
            trR = -1.0
            if not np.isclose(m[2, 2], -1.0, rtol=1e-3):
                k = 1.0/(np.sqrt(2.0*(1.0+m[2][2]))) * \
                    np.array([m[0][2], m[1][2], 1.0+m[2][2]])
            elif not np.isclose(m[1, 1], -1.0, rtol=1e-3):
                k = 1.0/(np.sqrt(2.0*(1.0+m[1][1]))) * \
                    np.array([m[0][1], 1.0+m[1][1], m[2][1]])
            else:
                k = 1.0/(np.sqrt(2.0*(1.0+m[0][0]))) * \
                    np.array([1.0+m[0][0], m[1][0], m[2][0]])
        else:
            a = np.arccos((trR-1)/2)
            if(np.isclose(a, 0.0, rtol=1e-2)):
                a = 0.0
                return cls(t=t)
            #print('regner ut k2')
            k = (1.0/(2.0*np.sin(a))) * \
                (np.array([m[2][1]-m[1][2], m[0][2]-m[2][0], m[1][0]-m[0][1]]))
        return cls(a, k, t)

    @classmethod
    def from_vecs(cls, r=np.array([0, 0, 0]), t=np.array([0, 0, 0])):
        a = np.linalg.norm(r)
        k = r / a
        return cls(a, k, t)

    @classmethod
    def from_kuka_pose(cls, x, y, z, a, b, c):
        '''derivation for zyx euler angles
        found in modern robotics appendix B.1'''
        Ca = np.cos(a)
        Cb = np.cos(b)
        Cc = np.cos(c)
        Sa = np.sin(a)
        Sb = np.sin(b)
        Sc = np.sin(c)
        m = np.array([[Ca*Cb, Ca*Sb*Sc-Sa*Cc, Ca*Sb*Cc+Sa*Sc, x],
                      [Sa*Cb, Sa*Sb*Sc+Ca*Cc, Sa*Sb*Cc-Ca*Sc, y],
                      [-Sb, Cb*Sc, Cb*Cc, z],
                      [0, 0, 0, 1]])
        return cls.from_matrix(m)

    def to_kuka_pose(self):
        m = self.rot_matrix
        beta = np.arctan2(-m[2][0], np.sqrt(m[0][0]**2+m[1][0]**2))
        alpha = np.arctan2(m[1][0], m[0][0])
        gamma = np.arctan2(m[2][1], m[2][2])
        return np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)

    def to_mm(self):
        transmm = self.t*1000
        angle, axis = self.angle_axis()
        return HTransf(angle, axis, transmm)

    def __str__(self):
        # defines what is printed when invocing the print() method on the object
        return "{}\n".format(self.matrix)

    def __repr__(self):
        return 'HTransf object'

    def __matmul__(self, other):
        # overloads the matrix multiplication operator '@'
        if isinstance(other, np.ndarray):
            if other.shape == (3,):
                return self.rot_matrix @ other
            return self.matrix @ other
        return HTransf.from_matrix(self.matrix @ other.matrix)

    def __add__(self, other):
        # note that for addition and subptraction the result is not in SE(3)
        if isinstance(other, np.ndarray):
            return self.matrix+other
        return self.matrix+other.matrix

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return self.matrix-other
        return self.matrix-other.matrix

    def tr(self):
        return np.trace(self.matrix)

    def tr_rot(self):
        return np.trace(self.rot_matrix)

    def angle_axis(self):
        '''
        returns the angle-axis representation of the rotation matrix
        '''
        return (self.angle, self.axis)

    def rvec(self):
        a, k = self.angle_axis()
        return a*k

    def getTranslation(self):
        return self.t

    def inv(self):
        '''returns a HTransf object representing the inverse 
        Homogeneous transformation matrix
        of the calling object'''
        return HTransf.from_matrix(np.linalg.inv(self.matrix))

    def getMatrix(self):
        return self.matrix

    def confirm_SO3(self):
        '''
        returns true if the rotation is member of the SO(3) group, false otherwise.
        '''
        det = np.linalg.det(self.rot_matrix)
        rotTransp = self.rot_matrix.T
        rotInv = np.linalg.inv(self.rot_matrix)
        if (np.isclose(det, 1.0, atol=1e-3) and np.isclose(np.sum(rotTransp-rotInv), 0.0, atol=1e-3)):
            return True
        else:
            return False


if __name__ == "__main__":
    a = np.ones((3, 3))
    print(a)
    print(np.trace(a))
