"""Compute linear extensions of four dimensional Boolean lattice for various interaction functions

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/21/21; Last revision: 4/21/21
"""

import numpy as np
from itertools import product
import pickle


def sampler(nSample, dimension):
    """Sample parameters in [0,10]^(2N) uniformly at random"""
    return 10 * np.squeeze(np.random.rand(nSample, 2 * dimension))


# functions for computing linear extensions
class ACLEP:
    """An instance of ACLEP associated to linear extensions of a Boolean lattice."""

    def __init__(self, interaction, dimension, init_samples=1000, max_extensions=np.inf):
        self.Interaction = interaction
        self.Dimension = dimension
        self.Sampler = sampler
        self.MaxExtensions = max_extensions
        self.Extensions = set()
        self.Samples = 0
        self.sample_linear_extensions(init_samples)

    def __repr__(self):
        """Print solutions to the screen"""
        sols = sorted(list(self.Extensions))
        sol_strings = [str(sol) + '\n' for sol in sols]
        return ''.join(sol_strings)

    def __len__(self):
        """Return the number of linear extensions found"""
        return len(self.Extensions)

        # methods related to sampling and finding linear extensions

    def boolean_index(self, k):
        """Return a tuple of length N with the binary digits of k in big endian order"""
        idx = [0 for j in range(self.Dimension)]
        bin_digits = [int(d) for d in bin(k)[2:]]
        idx[-len(bin_digits):] = bin_digits
        return tuple(idx)

    def prec(self, p, q):
        """Returns true if p < q in the Boolean lattice partial order. Returns false if q < p or if they are
        incomparable """
        p_bits = np.sum(self.boolean_index(p))
        q_bits = np.sum(self.boolean_index(q))
        return p_bits < q_bits

    def delta_to_sum(self, parameter):
        """Input a vector in R^(2N) of the form (L_1,...,L_N, delta_1,..., delta_N).
        Output a vector in R^(2N) of the form (L_1,...,L_N, L_1 + delta_1,...,L_N + delta_N)"""
        parm = parameter.copy()
        parm[self.Dimension:] += parameter[:self.Dimension]
        return parm

    def dual_extension(self, images):
        """Returns the dual linear extension with respect to the N dimensional Boolean lattice"""

        if len(images) == 1:
            return images

        else:
            this_idx = images[0]
            i = 1
            while not self.prec(this_idx, images[i]):  # p is not less than q in the Boolean partial order
                i += 1
            return images[i - 1::-1] + self.dual_extension(images[i:])

    def parameter_combinations(self, parameter):
        """Input a vector in R^(2N) of the form (L_1,...,L_N, delta_1,..., delta_N).
        Output an array with 2^N rows and N columns corresponding to the parameters defining a complete list of
        interaction polynomials.

        Example: Input: (0, 0, 0, 1, 1, 1) outputs the Boolean indices:
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]"""
        U_parameter = self.delta_to_sum(parameter)  # switch from (ell, delta) to (L, U) coordinates
        eval_pairs = (U_parameter[[j, self.Dimension + j]] for j in range(self.Dimension))
        return np.array(list(product(*eval_pairs)))

    def total_order(self, parameter):
        """Retrieve the total order of a given parameter with respect to a given interaction function"""
        parm_combo = self.parameter_combinations(parameter)
        evals = [self.Interaction(parm) for parm in parm_combo]
        return tuple(np.argsort(evals))

    def sample_linear_extensions(self, nSample):
        """Randomly sample parameters and return all unique linear extensions witnessed by the sample."""
        parameters = self.Sampler(nSample, self.Dimension)
        linear_extensions = set(map(lambda parm: self.total_order(parm), parameters))
        self.Extensions = linear_extensions.union(self.Extensions)
        self.Samples += nSample

    def continuous_sample(self):
        """Continuously sample linear extensions until no new ones are found anymore"""
        nIter = 0
        while nIter < 5000 and len(self) < self.MaxExtensions:
            nExtensions = len(self)
            self.sample_linear_extensions(10000)
            if len(self) > nExtensions:  # new extensions found
                nIter = 0  # reset the counter
            else:
                nIter += 1

    # methods related to comparing solutions for several interactions
    def union(self, aclep_instance):
        """Return the union of two ACLEP instances"""
        return self.Extensions.union(aclep_instance.Extensions)

    def intersection(self, aclep_instance):
        """Return the intersection of two ACLEP instances"""
        return self.Extensions.intersection(aclep_instance.Extensions)


# 3 variable interactions
def x_y_z(u):
    """Evaluate interaction function: (x + y + z) on R^3"""
    return np.sum(u)


def xz_yz(u):
    """Evaluate interaction function: (x + y)z on R^3"""
    return np.sum(u[:-1]) * u[-1]


def xy_z(u):
    """Evaluate interaction function: xy + z on R^3"""
    return np.prod(u[:-1]) + u[-1]


# 4 variable interactions
def x_y_z_w(u):
    """Evaluate interaction function: (x + y + z + w) on R^4"""

    return np.sum(u)


def xw_yw_zw(u):
    """Evaluate interaction function: (x + y + z)w on R^4"""
    return np.sum(u[:-1]) * u[-1]


def xz_xw_yz_yw(u):
    """Evaluate interaction function: (x + y)(z + w) on R^4"""
    return np.sum(u[0:2]) * np.sum(u[2:])


def xzw_yzw(u):
    """Evaluate interaction function: (x + y)zw on R^4"""
    return np.sum(u[0:2]) * u[2] * u[3]


# # initialize computations
# psdSolutions = dict()
# psdSolutions['x+y+z'] = ACLEP(x_y_z, 3, max_extensions=12)  # (3) has 12 extensions
# psdSolutions['(x+y)z'] = ACLEP(xz_yz, 3, max_extensions=20)  # (2, 1) has 20 extensions
# psdSolutions['xy+z'] = ACLEP(xy_z, 3, max_extensions=20)  # 20 extensions
# psdSolutions['x+y+z+w'] = ACLEP(x_y_z_w, 4, max_extensions=336)  # (4) has 336 extensions
# psdSolutions['(x+y)zw'] = ACLEP(xzw_yzw, 4, max_extensions=1334)  # (2, 1, 1) has 1,344 extensions
# psdSolutions['(x+y)(z+w)'] = ACLEP(xz_xw_yz_yw, 4, max_extensions=5344)  # (2, 2) has 5,344 extensions
# psdSolutions['(x+y+z)w'] = ACLEP(xw_yw_zw, 4, max_extensions=3084)  # (3, 1) has 3,084 extensions
# print([len(aclep) for aclep in psdSolutions.values()])  # print current solution counts
#
#
# # save initial solutions
# fileHandler = open('psd_solutions', 'wb')
# pickle.dump(psdSolutions, fileHandler)
# fileHandler.close()

# read in current solutions
fileReader = open('psd_solutions', 'rb')
psdSolutions = pickle.load(fileReader)
fileReader.close()
print([len(aclep) for aclep in psdSolutions.values()])  # print current solution counts

# sample some more
for aclep in psdSolutions.values():
    aclep.continuous_sample()
print([len(aclep) for aclep in psdSolutions.values()])  # print new solution counts
print([aclep.Samples for aclep in psdSolutions.values()])  # print new solution counts

# write updated solutions
fileWriter = open('psd_solutions', 'wb')
pickle.dump(psdSolutions, fileWriter)
fileWriter.close()