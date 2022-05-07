#!/usr/bin/env python3

import math
import sys
import numpy as np

from pyscf import gto
from pyscf.lib import logger


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

def kinetic_energy(density, kin):
    return np.sum(np.sum(density*kin))

def density_rmsd(den, old_den):
    # sqrt{sum_{ij} (D_{ij} - Dnew_{ij})^2}
    diff = den - old_den
    rmsd = np.sum(np.sum(diff*diff))
    return np.sqrt(rmsd)


def print_matrix(mat, log=None):
    WRITE_OUT = print
    if log is not None:
        write_out = log.info

    s = "   "
    for i in range(len(mat)):
        s += f"{i:^10}"

    WRITE_OUT(s)

    for i, r in enumerate(mat):
        s = f"{i:<3}"
        for c in r:
            s += f"{c:9.5f} "
        
        WRITE_OUT(s)


def _write(mol, vec, log=None, atmlst=None):
    WRITE_OUT = print
    if log is not None: WRITE_OUT = log.info

    WRITE_OUT('         x                y                z')
    if atmlst is None:
        atmlst = range(mol.natm)

    for k, ia in enumerate(atmlst):
        WRITE_OUT('%d %s  %15.10f  %15.10f  %15.10f' % (ia, mol.atom_symbol(ia), vec[k,0], vec[k,1], vec[k,2]))


# offers canonical diagonalization
def diagonalize(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # I hate numpy for not automatically sorting eigenvalues/eigenvectors by smallest value
    # this fixes this so that we don't have issues later on...so stupid
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def rhf(mol, e_thr=1-9, dens_thr=1e-9, max_scf=125, verbose=logger.NOTE):
    log = logger.new_logger("rhf", verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    log.info("Input Geometry")
    _write(mol, mol.atom_coords(), log=log)

    log.info("\nComputing 1e and 2e integrals...")
    mol.build()
    t1 = log.timer("Computing 1e and 2e integrals", *t0)

    kin = mol.intor('int1e_kin', aosym='s1')
    vnuc = mol.intor('int1e_nuc', aosym='s1')
    overlap = mol.intor('int1e_ovlp', aosym='s1')
    eri = mol.intor('int2e', aosym='s1')
    
    log.debug("Generating Initial Matrices")
    core = construct_core(kin, vnuc)
    lam, L = diagonalize(overlap)
    lam = 1.0/np.sqrt(lam)
    sym_ortho = L.dot(np.diag(lam).dot(L.T))
    sym_ortho_t = sym_ortho.T
    t2 = log.timer("Generating Initial Matrices", *t1)

    if verbose >= logger.INFO:
        log.info("\n--- [e Kinetic Energy Matrix] ---")
        print_matrix(kin, log=log)
        log.info("\n--- [E-Nuc Attraction Matrix] ---")
        print_matrix(vnuc, log=log)
        log.info("\n------- [Overlap Matrix] --------")
        print_matrix(overlap, log=log)
        log.info("\n--------- [Core Matrix] ---------")
        print_matrix(core, log=log)
        log.info("\n-------- [S^-1/2 Matrix] --------")
        print_matrix(sym_ortho, log=log)

    # Initial guess
    log.debug("Generating Initial Guess")
    t2 = (logger.process_clock(), logger.perf_counter())
    fock = sym_ortho_t.dot(core.dot(sym_ortho))
    e, C_prime = diagonalize(fock)
    C = sym_ortho.dot(C_prime)

    density = construct_density(C, mol.nelectron)
    t3 = log.timer("Generating Initial Guess", *t2)

    if verbose >= logger.INFO:
        log.info("\n--------- [Initial MOs] ---------")
        print_matrix(C, log=log)
        log.info("\n------- [Initial Density] -------")
        print_matrix(density, log=log)

    electron_energy = electronic_energy(density, core, core)
    log.info(f"\nInitial Electronic Energy: {electron_energy:.10f}")
    nuclear_energy = mol.energy_nuc()
    log.info(f"Nuclear-Nuclear Repulsion: {nuclear_energy:.10f}")

    log.debug("\nStarting SCF procedure...")
    log.info("\n******** SCF flags ********")
    log.info(f"Energy Threshold: {e_thr}")
    log.info(f"Density Threshold: {dens_thr}")
    log.info(f"Max SCF: {max_scf}")
    
    converged = False
   
    t3 = (logger.process_clock(), logger.perf_counter())
    for cycle in range(max_scf):
        fock = core + construct_G(eri, density)
        tfock = sym_ortho_t.dot(fock.dot(sym_ortho))

        e, C_prime = diagonalize(tfock)
        C = sym_ortho.dot(C_prime)

        new_density = construct_density(C, mol.nelectron)
        
        new_energy = electronic_energy(new_density, core, fock)

        rmsd = density_rmsd(new_density, density)
        delta_e = new_energy - electron_energy
        electron_energy = new_energy

        if np.abs(rmsd) < dens_thr and np.abs(delta_e) < e_thr:
            converged = True

        density = new_density
        
        if cycle == 0 or verbose >= logger.DEBUG:
            log.info(
                f"\n{'Iter':^5} {'E (elec)':^20} {'E Tot':^20} {'Delta E':^20} {'RMS D': ^20} {'Converged?': ^10}")
            log.info("-------------------------------------------------------------------------------------------------------")

        log.info(f"{cycle:^5} {new_energy:18.13f} {new_energy+nuclear_energy:18.13f} {delta_e:18.13f} {rmsd:18.13f} {'Yes' if converged else 'No':>10}")
        
        if verbose >= logger.DEBUG:
            log.info("\n---------- [MOs Coeff] ----------")
            print_matrix(C, log=log)
            log.info("\n----------- [Density] -----------")
            print_matrix(density, log=log)

        t3 = log.timer("SCF iteration %d" % cycle, *t3)
        
        if converged:
            break
        

    if converged:
        log.info("\n            ******************************************************")
        log.info("            *                     SUCCESS                        *")
        log.info(f"            *           SCF CONVERGED AFTER {cycle:4d} CYCLES          *")
        log.info("            ******************************************************")

    else:
        log.warn(f"SCF did not Converge in {cycle} iterations")
        log.warn("Try increasing SCF iterations...")

    if verbose >= logger.INFO:
        log.info("\n---------- [Final MOs] ----------")
        print_matrix(C, log=log)
    
    log.note('\n-------------------------------------------')
    print(f'FINAL SCF ENERGY     = {electron_energy+nuclear_energy:18.13f}')
    log.note('-------------------------------------------')
   
    t0 = log.timer("RHF", *t0)
    return electron_energy+nuclear_energy


def main():
    mol = gto.M(atom='''O 0 -0.14322 0; H 1.63803 1.13654 0; H -1.63803 1.13654 0''',
                basis='sto3g', unit="Bohr")

    e_conv = 1e-9
    max_cycle = 200
    
    print("********* Flags *********")
    print(f"conv_tol = {e_conv}")
    print(f"max_cycl = {max_cycle}")
    print("\n******** MRH-SCF ********")
    my_energy = rhf(mol, e_thr=e_conv, max_scf=max_cycle, verbose=logger.INFO)

    print("\n********* PySCF *********")

    pyscf_rhf = mol.RHF(conv_tol=e_conv)
    pyscf_rhf.max_cycle = max_cycle 

    diff = abs(my_energy-pyscf_rhf.kernel())
    print(f"\nEnergy Difference = {diff}") 

if __name__ == "__main__":
    main()

