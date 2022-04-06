#!/usr/bin/env python3

import math
import numpy as np


def fact2(n: int):
    return math.prod(range(n, 0, -2))


# McMurchie-Davidson Scheme and Hermite Gaussians coefficients
def E(i, j, t, R, alpha, beta):
    p = alpha + beta
    q = alpha * beta / p

    if t < 0 or (t > (i + j)):
        return 0

    elif i == 0 and j == 0 and t == 0:
        return np.exp(-q * R * R)

    elif j == 0:
        return E(i - 1, j, t - 1, R, alpha, beta) / (2 * p) - (q * R * E(i - 1, j, t, R, alpha, beta)) / alpha + (
                t + 1) * E(i - 1, j, t + 1, R, alpha, beta)

    else:
        return E(i, j - 1, t - 1, R, alpha, beta) / (2 * p) + (q * R * E(i, j - 1, t, R, alpha, beta)) / beta + (
                t + 1) * E(i, j - 1, t + 1, R, alpha, beta)


def primitive_gto_overlap(a, shell_a, center_a, b, shell_b, center_b):
    s = [E(shell_a[i], shell_b[i], 0, center_a[i] - center_b[i], a, b) for i in range(3)]
    return s[0] * s[1] * s[2] * np.power(np.pi / (a + b), 1.5)


class contracted_gto:

    def __init__(self, exponents, contractions, shell=(0, 0, 0), center=np.array([0, 0, 0])):
        self.exponents = exponents
        self.contractions = contractions
        self.shell = shell
        self.center = center
        self.norms = None
        self.normalize()

    def normalize(self):
        # Normalize primitive gaussians
        # Given <G_i|G_i> = (2i-1)!!/(4a)^i * sqrt(pi/2a), then we can write
        # <G_ijk|G_ijk> = (2i-1)!!(2j-1)!!(2k-1)!!/(2^(2+i+j+k+1.5)a^(i+j+k+1.k)) *pi^(1.5)
        # Here we take the reciprical...
        L = sum(self.shell)
        self.norm = np.sqrt(
            np.power(2, 2 * L + 1.5) * np.power(self.exponents, L + 1.5) / fact2(2 * self.shell[0] - 1) / fact2(
                2 * self.shell[1] - 1) / fact2(2 * self.shell[2] - 1) / np.power(np.pi, 1.5)
        )

        # l, m, n = self.shell
        # pre = np.power(np.pi, 1.5)* fact2(2*l-1)*fact2(2*m-1)*fact2(2*n-1)/np.power(2.0,L)
        #
        # norm = 0.0
        # for i in range(len(self.exponents)):
        #     for j in range(len(self.exponents)):
        #         norm += self.norm[i]*self.norm[j]*self.contractions[i]*self.contractions[j]/np.power(self.exponents[i]+self.exponents[j], L+1.5)
        #
        # norm *= pre
        # norm = np.power(norm, -0.5)
        # for i in range(len(self.exponents)):
        #     self.contractions[i] *= norm

        # could be done cheaper...but this should work in all honesty

        norm = np.sqrt(self.overlap(self))
        self.contractions = [c / norm for c in self.contractions]

    def overlap(self, other_gto):
        s = 0.0
        for i, ci in enumerate(self.contractions):
            for j, cj in enumerate(other_gto.contractions):
                s += self.norm[i]*other_gto.norm[j]*ci * cj * primitive_gto_overlap(self.exponents[i], self.shell,
                                                     self.center,
                                                     other_gto.exponents[j],
                                                     other_gto.shell,
                                                     other_gto.center)
        return s


def read_basis_file(file_name: str):
    basis_set = {}

    with open(file_name, 'r') as basis_file:
        begin_read = False
        orbitals = []
        exponents = []
        contractions = []
        atom = ""

        def add_basis_elements():
            for index, orbital in enumerate(orbitals):
                if orbital not in basis_set[atom].keys():
                    basis_set[atom][orbital] = []

                basis_set[atom][orbital].append((exponents, [c[index] for c in contractions]))

        for line in basis_file:
            line = line.strip("\n")
            if not begin_read and line.startswith("BASIS \"ao basis\" SPHERICAL PRINT"):
                line = line.split("\"")
                basis_set["title"] = line[1]
                # line_info = [l for l in line[2].split(" ") if l]
                begin_read = True

            elif not line or line.startswith("#"):
                continue

            elif line.startswith("END"):
                break

            elif begin_read:
                line = line.split(" ")
                line = [l for l in line if l]
                try:
                    vals = [float(l) for l in line]
                    line = vals

                except:
                    pass

                if type(line[0]) is str:
                    if atom != "":
                        add_basis_elements()

                        exponents = []
                        contractions = []
                        atom = line[0]
                        if atom not in basis_set.keys():
                            basis_set[atom] = {}

                        orbital_types = line[1]
                        orbitals = [c for c in orbital_types]

                    else:
                        atom = line[0]
                        basis_set[atom] = {}
                        orbital_types = line[1]
                        orbitals = [c for c in orbital_types]

                else:
                    exponents.append(line[0])
                    contractions.append(line[1:])

        add_basis_elements()

    return basis_set
