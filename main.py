from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units
from ase.io import Trajectory, write
import ase
import torchani
import numpy as np


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
    elif random_num < np.exp(-dE/(units.kB * T)):
        return True
    return False

def run_MC(atoms,T_max,steps,dr):
    total_moves = 0
    total_accepted = 0
    total_rejected = 0
    T = T_max
    traj = Trajectory('mc_opt.traj', 'w', atoms)
    f_ = open('mc_steps.txt','w')
    for step in range(steps):
        if total_moves % 100 == 0 and step > 1:
            acceptance_rate = total_accepted / total_moves
            # if the acceptance rate is too low, reduce displacement
            # if its too high, increase displacement
            # adjust every 100 steps
            if acceptance_rate < 0.3:
                dr = dr*0.9
            if acceptance_rate > 0.5:
                dr = dr*1.1
        # simulated annealing to reduce temperature linearly
        # and an early stop for the run if we get too cold (20K for now)
        if T < 20:
            break
        T = T_max * (1 - (step + 1)/steps)
        E_i = atoms.get_potential_energy()
        starting_coordinates = atoms.get_positions()
        displacement = displace(atoms,dr)
        new_coordinates = starting_coordinates + displacement
        atoms.set_positions(new_coordinates)
        E_f = atoms.get_potential_energy()
        if metropolis(E_i,E_f,T):
            f_.write(f"accepted: {E_f}\n")
            total_moves+=1
            total_accepted+=1
            traj.write()
        else:
            f_.write(f"rejected\n")
            atoms.set_positions(starting_coordinates)
            total_moves+=1
            total_rejected+=1
        if abs(E_f - E_i) < 1e-6 and step > 1:
            f_.write(f"converged\n")
            break
    f_.write(f'total steps: {total_moves}, total accepted: {total_accepted}, total rejected: {total_rejected}\n')
    traj.close()
    traj = Trajectory('mc_opt.traj','r')
    write('mc_opt_traj.xyz',traj)
    write('molecule_final.xyz',atoms,format='xyz')
    f_.close()
    

    
def main():
    # ASE energy units in eV
    calculator = torchani.models.ANI2x().ase()

    atoms = ase.io.read('molecule.xyz',format='xyz')
    atoms.set_calculator(calculator)

    # move atoms by at most 0.01 angstroms
    dr = 0.01
    steps = 1000000
    # temp in kelvin
    T = 10000
    run_MC(atoms,T,steps,dr)




if __name__ == '__main__':
    main()

