#!/usr/bin/env python3

import numpy as np
import sys

from gto import *

ANG_TO_BOHR = 1.88973

def read_xyz_file(file_name: str):
    lines = []
    with open(file_name, 'r') as xyz_file:
        for line in xyz_file:
            line = line.strip("\n")
            lines.append(line)

    num_atoms = int(lines[0])

    structure = []
    for line in lines[2:2+num_atoms]:
        line = line.split(" ")
        element = line[0]
        coordinate = [ANG_TO_BOHR*float(l) for l in line[1:]]
        coordinate = np.asarray(coordinate)
        structure.append((element, coordinate))

    return structure

def generate_aos(structure, basis_set):
    aos = []
    for atom in structure:
        element = atom[0]
        coordinate = atom[1]
        for orbital_type in basis_set[element]:
            if orbital_type == "S":
                possible_shells = [(0,0,0)]
            elif orbital_type == "P":
                possible_shells = [(1, 0, 0), (0, 1, 0), (0,0,1)]
            else:
                print("ERRORROROROR")
                sys.exit(1)

            for contracted_ao in basis_set[element][orbital_type]:
                for shell in possible_shells:
                    aos.append(contracted_gto(contracted_ao[0], contracted_ao[1],
                                          shell=shell, center=coordinate))

    return aos

def construct_overlap_matrix(aos):
    s = np.zeros((len(aos),len(aos)))
    for i in range(len(aos)):
        for j in range(i,len(aos)):
            s[i][j] = aos[i].overlap(aos[j])
            s[j][i] = s[i][j]

    return s

def main():
    basis_set = read_basis_file("basis_file")
    structure = read_xyz_file("tmp.xyz")

    aos = generate_aos(structure, basis_set)

    overlap = construct_overlap_matrix(aos)
    print(overlap)

if __name__ == "__main__":
    main()
