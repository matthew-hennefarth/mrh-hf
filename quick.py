#!/usr/bin/env python3

from pyscf import gto
import numpy as np

def construct_core(kin, nuc):
    return kin + nuc

def construct_G(two_e_int, density):
    num_aos = two_e_int.shape[0]
    
    G = np.zeros((num_aos, num_aos))

    for mu in range(num_aos):
        for nu in range(num_aos):
            e_e_repulsion_component = 0
            for lam in range(num_aos):
                for sigma in range(num_aos):
                    dens = density[lam][sigma]
                    correlation = two_e_int[mu][nu][sigma][lam]
                    exchange = two_e_int[mu][lam][sigma][nu]

                    e_e_repulsion_component += dens * (correlation - 0.5 * exchange)

            G[mu][nu] = e_e_repulsion_component

    return G

def construct_density(mos):
    density = np.zeros((len(mos), len(mos)))
    for i in range(len(mos)):
        for j in range(i,len(mos)):
            s = 0
            for k in range(len(mos[i])):
                s += mos[i][k] * mos[j][k]

            density[i][j] = 2*s
            #density[i][j] = 2*mos[i].dot(mos[j])
            density[j][i] = density[i][j]
    
    return density

def main():
    mol = gto.M(atom = '''O 0 0 0; H 0 0.751 -0.465; H 0 -0.751 -0.465''',
            basis = 'sto-3g')
    mol.build()
    mol.spin = 1
    mol.charge = 0
    
    kin = mol.intor('int1e_kin')
    vnuc = mol.intor('int1e_nuc')
    overlap = mol.intor('int1e_ovlp')
    eri = mol.intor('int2e', aosym='s1')

    initial_mos = np.array([
            np.array([0.994123, 0.025513, 0, 0, -0.002910, -0.005147, -0.005147]),
            np.array([-0.232461, 0.833593, 0,0,-0.140863, 0.155621, 0.155621]),
            np.array([0,0,0,0.607184, 0, 0.444175, -0.444175]),
            np.array([-0.107246, 0.556639, 0, 0, 0.766551, -0.285923, -0.285923]),
            np.array([0,0,0,0,0,0,0]),
            np.array([0,0,0,0,0,0,0]),
            np.array([0,0,0,0,0,0,0])
            ])

    density = construct_density(initial_mos)

    s,U = np.linalg.eigh(overlap)
    s = [1.0/np.sqrt(i) for i in s]
    s = np.array(s)
    s = np.diag(s)
    
    X = np.matmul(U, s)
    X_adj = np.conj(X).T

    core = construct_core(kin, vnuc)    

    for i in range(10):
        G = construct_G(eri, density)
        fock = core + G
        t_fock = np.matmul(X_adj, np.matmul(fock, X))

        e,c = np.linalg.eig(t_fock)

        c = np.matmul(X,c)
        
        new_density = construct_density(c.T)

        # check to see if converged

    print(e)
#   print(c)



if __name__ == "__main__":
    main()
