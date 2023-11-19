# SIR particle dynamics

This is a simple simulation of scattering particles that can be used to e.g. simulate the SIR (susceptible-infected-recovered) dynamics and visualize it in runtime by matplotlib's FuncAnimation.

The backbone of the simulation is the class 'particle' that mainly implements the movement of a given particle and its pair-wise interaction with another particle via elastic collision.
The class 'box_ensemble' provides a wrapper to simulate all particles simultaneously and defines a grid map to reduce the time complexity of computing collision events. 
This is done by tracking a 2D array with labels of different particles at different positions, s.t. upon searching through the candidates for collision events, a particle only has to look in its immediate neighborhood 
(this is often used in the notorious [boids algorithm](https://en.wikipedia.org/wiki/Boids).

The main parameters of the infection dynamics can be changed in the particle class, whereas the box_ensemble class takes the number of particles and their sizes as input.
For example the SIR dynamics with varying particles sizes could look like this:

https://github.com/borya-polovnikov/SIR_particle_dynamics/assets/147932035/24605222-f91e-45a9-a6f4-0463a9128757


The reaction dynamics of individual particles can also be changed inside the particle class, e.g. to implement the [Diffusive Epidemic Process](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.078302):

https://github.com/borya-polovnikov/SIR_particle_dynamics/assets/147932035/e9d61abf-d92d-4145-a966-75bfaff8d2d1
