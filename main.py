#!/usr/bin/env python3

import numpy as np

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
        coordinate = np.asarray(line[1:])
        structure.append((element, coordinate))

    return structure

def generate_aos(structure, basis_set):
    # ((l_x,l_y,l_z), center, (alpha's), (contractions))
    aos = []
    for atom in structure:
        element = atom[0]
        coordinate = atom[1]



def main():
    basis_set = read_basis_file("basis_file")
    structure = read_xyz_file("tmp.xyz")

    aos = generate_aos(structure, basis_set)


if __name__ == "__main__":
    main()
