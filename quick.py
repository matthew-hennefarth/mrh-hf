#!/usr/bin/env python3

import math
import numpy as np

from pyscf import gto

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

                    e_e_repulsion_component += dens * (2*correlation - exchange)

            G[mu][nu] = e_e_repulsion_component

    return G

def construct_density(mos, n_elecs):
    density = np.zeros((len(mos), len(mos)))
    occupied = int(n_elecs/2)
    
    for mu in range(len(mos)):
        for nu in range(mu,len(mos)):
            s = 0
            for k in range(occupied):
                s += mos[mu][k] * mos[nu][k]

            density[mu][nu] = s
            density[nu][mu] = s
    
    return density

def electronic_energy(density, core, fock):
    rows = len(density)
    energy = 0
    for mu in range(rows):
        for nu in range(rows):
            energy += density[mu][nu]*(core[mu][nu] + fock[mu][nu])

    return energy


def density_rmsd(den, old_den):
    rmsd = 0
    diff = den - old_den
    for row in diff:
        for i in row:
            rmsd += i*i

    rmsd = np.sqrt(rmsd)
    return rmsd

def print_matrix(mat):
    s = "   "
    for i in range(len(mat)):
        s += f"{i:^10}"
    
    print(s)
    
    for i,r in enumerate(mat):
        s = f"{i:<3}"
        for c in r:
            s += f"{c:9.5f} " 
        print(s)

# offers canonical diagonalization
def diagonalize(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    return eigenvalues, eigenvectors

def rhf(mol, e_thr=-6, dens_thr=-6, max_scf=125):
    mol.build()

    nuclear_energy = mol.energy_nuc()

    kin = mol.intor('int1e_kin', aosym='s1')
    print("---- [Kinetic Energy Matrix] ----")
    print_matrix(kin)
    
    vnuc = mol.intor('int1e_nuc', aosym='s1')
    print("\n")
    print("--------- [VNuc Matrix] ---------")
    print_matrix(vnuc)
    
    overlap = mol.intor('int1e_ovlp', aosym='s1')
    print("\n")
    print("------- [Overlap Matrix] --------")
    print_matrix(overlap)
    
    eri = mol.intor('int2e', aosym='s1')
    core = construct_core(kin, vnuc)

    print("\n")
    print("--------- [Core Matrix] ---------")
    print_matrix(core)

    lam, L = diagonalize(overlap)
    lam = 1.0/np.sqrt(lam)
    sym_ortho = L.dot(np.diag(lam).dot(L.T))
    sym_ortho_t = sym_ortho.T
    
    print("\n")
    print("-------- [S^-1/2 Matrix] --------")
    print_matrix(sym_ortho)

    # Initial guess
    fock = sym_ortho_t.dot(core.dot(sym_ortho)) 
    e,C_prime = diagonalize(fock)
    C = sym_ortho.dot(C_prime)
    print("\n")
    print("--------- [Initial MOs] ---------")
    print(e)
    print_matrix(C)

    density = construct_density(C, mol.nelectron)
    print("\n")
    print("------- [Initial Density] -------")
    print_matrix(density)

    energy = electronic_energy(density, core, core)
    print(f"\nInitial Electronic Energy: {energy:.10f}")
    print(f"Nuclear-Nuclear Repulsion: {nuclear_energy:.10f}")

    print("\nStarting SCF procedure...\n")
    converged = False
    for i in range(max_scf):
        if i == 0:
            print(f"{'Iter':^5} {'E (elec)':^20} {'E Tot':^20} {'Delta E':^20} {'RMS D': ^20} {'Converged?': ^10}")
            print("-------------------------------------------------------------------------------------------------------") 
        fock = core + construct_G(eri, density)
        tfock = sym_ortho_t.dot(fock.dot(sym_ortho))

        e, C_prime = diagonalize(tfock)
        C = sym_ortho.dot(C_prime)
        
        new_density = construct_density(C,mol.nelectron)
        new_energy = electronic_energy(new_density, core, fock)
       
        rmsd = density_rmsd(new_density, density) 
        delta_e = new_energy - energy
        energy = new_energy

        if np.abs(rmsd) < math.pow(10, dens_thr) and np.abs(delta_e) < math.pow(10,e_thr):
            converged = True

        density = new_density
        
        print(f"{i:^5} {new_energy:18.10f} {new_energy+nuclear_energy:18.10f} {delta_e:18.10f} {rmsd:18.10f} {'Yes' if converged else 'No':>10}")
        if converged:
            break

def main():
    mol = gto.M(atom = '''O 0 -0.14322 0; H 1.63803 1.13654 0; H -1.63803 1.13654 0''',
            basis = 'sto-3g', unit="Bohr")

    #mol = gto.M(atom = '''O 0 0 0.116; H 0 0.751 -0.465; H 0 -0.751 -0.465''',
    #        basis = 'sto-3g')

    #mol = gto.M(atom = '''
    #        C    3.402   0.773  -9.252 
    #        C    4.697   0.791  -8.909 
    #        H    2.933  -0.150  -9.521 
    #        H    2.837   1.682  -9.258 
    #        H    5.262  -0.118  -8.904 
    #        H    5.167   1.714  -8.641
    #        ''', basis = 'sto-3g')

    rhf(mol)


if __name__ == "__main__":
    main()
