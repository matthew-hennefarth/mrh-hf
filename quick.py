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

                    e_e_repulsion_component += dens * \
                        (2*correlation - exchange)

            G[mu][nu] = e_e_repulsion_component

    return G


def construct_density(mos, n_elecs):
    density = np.zeros((len(mos), len(mos)))
    occupied = int(n_elecs/2)

    for mu in range(len(mos)):
        for nu in range(mu, len(mos)):
            s = 0
            # we only go over occupied orbitals
            for k in range(occupied):
                s += mos[mu][k] * mos[nu][k]

            density[mu][nu] = s
            density[nu][mu] = s

    return density


def electronic_energy(density, core, fock):
    # sum_{mu,nu} D_{mu,nu}(Core_{mu,nu} + Fock_{mu,nu})
    return np.sum(np.sum(density*(core+fock)))


def density_rmsd(den, old_den):
    # sqrt{sum_{ij} (D_{ij} - Dnew_{ij})^2}
    diff = den - old_den
    rmsd = np.sum(np.sum(diff*diff))
    return np.sqrt(rmsd)


def print_matrix(mat):
    s = "   "
    for i in range(len(mat)):
        s += f"{i:^10}"

    print(s)

    for i, r in enumerate(mat):
        s = f"{i:<3}"
        for c in r:
            s += f"{c:9.5f} "
        print(s)

# offers canonical diagonalization


def diagonalize(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # I hate numpy for not automatically sorting eigenvalues/eigenvectors by smallest value
    # this fixes this so that we don't have issues later on...so stupid
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def rhf(mol, e_thr=-6, dens_thr=-6, max_scf=125):
    print("Computing 1e and 2e integrals...")
    mol.build()

    kin = mol.intor('int1e_kin', aosym='s1')
    print("\n---- [Kinetic Energy Matrix] ----")
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
    e, C_prime = diagonalize(fock)
    C = sym_ortho.dot(C_prime)
    print("\n")
    print("--------- [Initial MOs] ---------")
    print_matrix(C)

    density = construct_density(C, mol.nelectron)
    print("\n")
    print("------- [Initial Density] -------")
    print_matrix(density)

    energy = electronic_energy(density, core, core)
    print(f"\nInitial Electronic Energy: {energy:.10f}")
    nuclear_energy = mol.energy_nuc()
    print(f"Nuclear-Nuclear Repulsion: {nuclear_energy:.10f}")

    print("\nStarting SCF procedure...\n")
    print(f"Energy Threshold: 10e{e_thr}")
    print(f"Density Threshold: 10e{dens_thr}")
    print(f"Max SCF: {max_scf}\n")
    converged = False
    for cycle in range(max_scf):
        if cycle == 0:
            print(
                f"{'Iter':^5} {'E (elec)':^20} {'E Tot':^20} {'Delta E':^20} {'RMS D': ^20} {'Converged?': ^10}")
            print("-------------------------------------------------------------------------------------------------------")
        fock = core + construct_G(eri, density)
        tfock = sym_ortho_t.dot(fock.dot(sym_ortho))

        e, C_prime = diagonalize(tfock)
        C = sym_ortho.dot(C_prime)

        new_density = construct_density(C, mol.nelectron)
        new_energy = electronic_energy(new_density, core, fock)

        rmsd = density_rmsd(new_density, density)
        delta_e = new_energy - energy
        energy = new_energy

        if np.abs(rmsd) < math.pow(10, dens_thr) and np.abs(delta_e) < math.pow(10, e_thr):
            converged = True

        density = new_density

        print(f"{cycle:^5} {new_energy:18.10f} {new_energy+nuclear_energy:18.10f} {delta_e:18.10f} {rmsd:18.10f} {'Yes' if converged else 'No':>10}")
        if converged:
            break

    if converged:
        print(f"SCF Converged in {cycle} iterations")

    else:
        print(f"SCF did not Converge in {cycle} iterations")
        print("Try increasing SCF iterations...")

    print("\nFinal Data")
    print(f"Final Electronic Energy: {energy:.10f}")
    print(f"Final Total Energy: {energy+nuclear_energy:.10f}")

    print("\n")
    print("---------- [Final MOs] ----------")
    print_matrix(C)


def main():
    mol = gto.M(atom='''O 0 -0.14322 0; H 1.63803 1.13654 0; H -1.63803 1.13654 0''',
                basis='sto-3g', unit="Bohr")

    rhf(mol)


if __name__ == "__main__":
    main()
