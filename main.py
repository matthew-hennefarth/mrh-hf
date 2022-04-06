#!/usr/bin/env python3

import numpy as np

from gto import *

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
        coordinate = [float(l) for l in line[1:]]
        coordinate = np.asarray(coordinate)
        structure.append((element, coordinate))

    return structure

def generate_aos(structure, basis_set):
    aos = []
    for atom in structure:
        element = atom[0]
        coordinate = atom[1]
        for orbital_type in basis_set[element]:
            for contracted_ao in basis_set[element][orbital_type]:
                aos.append(contracted_gto(contracted_ao[0], contracted_ao[1],
                                          shell=(0, 0, 0), center=coordinate))

    return aos



def main():
    basis_set = read_basis_file("basis_file")
    structure = read_xyz_file("tmp.xyz")

    aos = generate_aos(structure, basis_set)

    for a in aos:
        for b in aos:
            print(a.overlap(b))

if __name__ == "__main__":
    main()
