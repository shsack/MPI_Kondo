from __future__ import division
from mpmath import *
import numpy as np

class bathCouplings:

    def __init__(self, density, N, Lambda, digits=300, datatype=np.float64):

        """constructs the couplings of the log transformed impurity model"""

        self.N = N
        self.Lambda = Lambda

        if density == 'const':

            self.t = np.array([((1. + self.Lambda ** -1.) * (1. - self.Lambda ** (-n - 1.)) * self.Lambda ** (-n / 2.)) /
                       (2. * sqrt(1. - self.Lambda ** (-2. * n - 1.)) * sqrt(1. - self.Lambda ** (-2. * n - 3.))) for n in
                       range(self.N - 1)])
            self.t = self.t.astype(datatype)

            self.eps = np.array([0] * (self.N))
            self.eps = self.eps.astype(datatype)

        else:

            # allows to use arbitrary precision given by set number of digits
            mp.dps = digits

            self.density = density

            x = [self.Lambda**-n for n in range(self.N)]

            gamma_plus = [sqrt(quad(self.density, [x[n+1], x[n]])) for n in range(self.N-1)]
            gamma_minus = [sqrt(quad(self.density, [-x[n], -x[n+1]])) for n in range(self.N-1)]

            norm_plus = [quad(self.density, [x[n+1], x[n]]) for n in range(self.N-1)]
            norm_minus = [quad(self.density, [-x[n], -x[n+1]]) for n in range(self.N-1)]

            xi_plus = [quad(lambda e: self.density(e) * e, [x[n+1], x[n]])/norm_plus[n] for n in range(self.N-1)]
            xi_minus = [quad(lambda e: self.density(e) * e, [-x[n], -x[n+1]])/norm_minus[n] for n in range(self.N-1)]
            xi_0 = quad(self.density, [-1, 1])

            t_square, self.t, self.eps = matrix([0]*self.N), matrix([0]*self.N), matrix([0]*self.N)

            self.eps[0] = quad(lambda e: self.density(e) * e, [-1, 1]) / xi_0

            t_square[0] = sum([((xi_plus[m] - self.eps[0])*gamma_plus[m])**2 +
                          ((xi_minus[m] - self.eps[0])*gamma_minus[m])**2 for m in range(self.N-1)]) / xi_0

            self.t[0] = sqrt(t_square[0])

            u, v = matrix([gamma_plus]) / sqrt(xi_0), matrix([gamma_minus]) / sqrt(xi_0)
            u.cols, v.cols = self.N-1, self.N-1
            u.rows, v.rows = self.N-1, self.N-1
            u += diag([0] + [1] * (self.N-2))
            v += diag([0] + [1] * (self.N-2))

            for m in range(self.N-1):
                u[1, m] = ((xi_plus[m] - self.eps[0]) * u[0, m]) / sqrt(t_square[0])
                v[1, m] = ((xi_minus[m] - self.eps[0]) * v[0, m]) / sqrt(t_square[0])

            for n in range(1, self.N-2):
                for m in range(self.N-1):
                    self.eps[n] += xi_plus[m] * (u[n, m]**2) + xi_minus[m] * (v[n, m]**2)
                    t_square[n] += ((xi_plus[m] * u[n, m])**2 + (xi_minus[m] * v[n, m])**2)

                t_square[n] = t_square[n] - t_square[n - 1] - self.eps[n]**2

                for m in range(self.N-1):

                    u[n + 1, m] = (((xi_plus[m] - self.eps[n]) * u[n, m]) - sqrt(t_square[n - 1]) * u[n - 1, m]) / sqrt(t_square[n])
                    v[n + 1, m] = (((xi_minus[m] - self.eps[n]) * v[n, m]) - sqrt(t_square[n - 1]) * v[n - 1, m]) / sqrt(t_square[n])
                self.t[n] = sqrt(t_square[n])

            # converts the arbitrary precision data typ to arbitrary data type, that allows to use numpy

            self.t = np.array([nstr(_, n=digits) for _ in self.t])
            self.t = self.t.astype(datatype)

            self.eps = np.array([nstr(_, n=digits) for _ in self.eps])
            self.eps = self.eps.astype(datatype)

