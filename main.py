from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units
import ase
import torchani
import numpy as np


# ASE energy units in eV
calculator = torchani.models.ANI1ccx().ase()
atoms = ase.Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
atoms.set_calculator(calculator)

# move atoms by at most 0.01 angstroms
dr = 0.01
steps = 500
# temp in kelvin
T = 600

def displace(atoms, dr):
    displacement = np.random.uniform(-1,1,size=atoms.positions.shape)
    for i, d in enumerate(displacement):
        norm = np.linalg.norm(d)
        if norm != 0:
            displacement[i] = (d/norm) * dr
        else:
            continue
    return(displacement)

def metropolis(E_i, E_f, T):
    random_num = np.random.rand()
    dE = E_f - E_i
    if dE < 0:
        return True
    elif random_num > np.exp(-dE/(units.kB * T)):
        return True
    return False

total_moves = 0
total_accepted = 0
total_rejected = 0

for step in range(steps):
    E_i = atoms.get_potential_energy()
    starting_coordinates = atoms.get_positions()
    displacement = displace(atoms,dr)
    new_coordinates = starting_coordinates + displacement
    atoms.set_positions(new_coordinates)
    E_f = atoms.get_potential_energy()
    if metropolis(E_i,E_f,T):
        print("accepted:", E_f)
        total_moves+=1
        total_accepted+=1
    else:
        print("rejected")
        atoms.set_positions(starting_coordinates)
        total_moves+=1
        total_rejected+=1
print(total_moves,total_accepted,total_rejected)
