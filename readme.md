Made a small code to do metropolis monte carlo optimization in ASE using ANI2x, but it can be changed to use any calculator
Takes in any .xyz in the directory, must be named molecule.xyz though
Runs monte carlo optimization, for 1 million steps starting at 10,000 kelvin
implemented simulated annealing with a linear decaying temperature
steps and starting temp can be changed of course
also has a dynamic step size, if acceptance rate is <30%, decrease step size by 10%, and if its above 50% increase it by 10%
Fun little project to test how to do monte carlo