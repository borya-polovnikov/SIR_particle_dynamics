# -*- coding: utf-8 -*-
"""
@author: Borislav Polovnikov
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class particle:
    '''
    Class representing a round particle.
    Base attributes are:
   	    position: [x,y]
   	    velocity: [vx, vy]
   	    radius: r
   	    mass: m
   	    
   	Each instance also has attributes associated with visualization, i.e. a matplotlib circle patch,
   	and attributes necessary to implement a reaction scheme (here: infection dynamics)
   	'''
    def __init__(self,  x=0, y=0, vx=0, vy=0, radius=3, mass=5):
        
        self._position = np.array([x,y])
        self._velocity = np.array([vx,vy])
        self._radius = radius
        self._mass = mass
        
        self._facecolor = "lightblue" # "lightblue" = healthy, "firebrick" = infected and "plum" = immune
        self._circle=patches.Circle(self._position,self._radius, fc=self._facecolor) # matplotlib artist, serves to represent the particle 
			
        self._infection_probability = 0.9 # probability of infection upon contact of healthy and infected particles
        self._mean_recovery_time = 10  # lifetime of sick particles, determined as the waiting time of an exponential distribution
        self._mean_immunization_time = 30 # time after which immunization is lost
        self._mobility_ratio = 1  # the mass of infected particles is multiplied by _mobility_ratio, resulting in lower mobility
    
    def set_position(self, pos_arg):
        self._position = pos_arg
        self._circle.set_center(pos_arg)
    def x(self):
        return self._position[0]
    def y(self):
        return self._position[0]
    
    def get_sick(self, infection_probability=None):
        if not infection_probability:
            infection_probability = self._infection_probability
        if self._facecolor == "lightblue": # if healthy & susceptible
            if (np.random.uniform() < infection_probability): # check probability of infection
                self._facecolor = "firebrick"
                self._circle.set_facecolor(self._facecolor)
                self._mass = self._mass*self._mobility_ratio
                self._velocity = self._velocity/self._mobility_ratio
                self._time_to_recovery = np.random.exponential(self._mean_recovery_time) # the time to recovery is exponentially distributed
    
    def recover(self):
        self._facecolor = "plum" # immune
        self._circle.set_facecolor(self._facecolor)
        self._mass = self._mass/self._mobility_ratio
        self._velocity = self._velocity*self._mobility_ratio
        self._time_to_susceptible = np.random.exponential(self._mean_immunization_time) # duration of immunity is exponentially distributed
    
    def move(self, dt):
        self._position += self._velocity*dt
        
        if self._facecolor == "firebrick": #if sick
            self._time_to_recovery -= dt
            if self._time_to_recovery < 0:
                self.recover()
        elif self._facecolor == "plum": #if immune
            self._time_to_susceptible -= dt
            if self._time_to_susceptible < 0:
                self._facecolor = "lightblue" # healthy and susceptible
                self._circle.set_facecolor(self._facecolor)
                
                
                
    def distance(self, other_particle): # distance to another particle
        return np.sqrt(np.dot(self._position-other_particle._position, self._position-other_particle._position ))
		
    def overlap(self, other_particle): # boolean value whether this particle overlaps with another one
        if self.distance(other_particle)<= (self._radius + other_particle._radius):
            return True
        else:
            return False
    
    
    def wall_reflection(self, xmin, xmax, ymin, ymax): # limit the space to a rectangular box with elastic walls
        
        if self._position[0] > xmax - self._radius:
            
            dt=(self._radius - (xmax - self._position[0]) )/self._velocity[0] # compute a time step to move back just before the collision happens
            self._position -= self._velocity*dt # move back just before the collision
            self._velocity[0] = -1*self._velocity[0] # swap velocity direction
            self.move(dt)

        if self._position[0] < xmin + self._radius:
            
            dt=(self._position[0] - xmin - self._radius)/self._velocity[0]
            self._position -= self._velocity*dt
            self._velocity[0] = -1*self._velocity[0]
            self.move(dt)
            
        if self._position[1] > ymax - self._radius:
            
            dt = (self._radius-(ymax-self._position[1]) )/self._velocity[1]
            self._position -= self._velocity*dt
            self._velocity[1] = -1*self._velocity[1]
            self.move(dt)

        if self._position[1] < ymin + self._radius:
            
            dt=(self._position[1] - ymin - self._radius)/self._velocity[1]
            self._position -= self._velocity*dt
            self._velocity[1] = -1*self._velocity[1]
            self.move(dt)
    
    
    def collision_update(self, other_particle): #update the velocities and health-states after two particles collide in an elastic collision
        
        if self.overlap(other_particle)==True:
            if (self._facecolor == "firebrick") and (other_particle._facecolor == "lightblue"): # this particle infects the other one
                other_particle.get_sick() 
            elif (other_particle._facecolor == "firebrick") and (self._facecolor == "lightblue"): # the other particle infects this one
                self.get_sick()	
            
            ###
            # compute the scattering in the relative coordinates
            ###
            M = self._mass + other_particle._mass # total mass
            P = self._mass*self._velocity + other_particle._mass*other_particle._velocity # total momentum must be conserved
			
            yf = other_particle._position - self._position # move back until the moment where the scattering has to be done
            yvelf = other_particle._velocity - self._velocity
            
            R = self._radius+other_particle._radius #total radius
            
            dt = np.abs(( np.dot(yf,yvelf) + np.sqrt(     np.max(   ( ( np.dot(yf,yvelf)**2   +(R**2-np.dot(yf,yf))*np.dot(yvelf,yvelf) ),0 )       )     )  )/np.dot(yvelf,yvelf))

            self._position -= self._velocity*dt #move back just before the collision
            other_particle._position -= other_particle._velocity*dt
			
			
            y = other_particle._position - self._position  # scattering computed in the two-body coordinates y=x2-x1 and X= (m1*x1 +m2*x2)/M
            yvel_before_collision  =   other_particle._velocity - self._velocity
            yvel_after_collision   =   yvel_before_collision   -  y*(   2*np.dot(y,yvel_before_collision)/np.dot(y,y)   )
			
            self._velocity = (P - other_particle._mass*yvel_after_collision)/M
            other_particle._velocity = (P  +  self._mass*yvel_after_collision)/M
            
            self.move(dt)
            other_particle.move(dt)



class box_ensemble:
    def __init__(self, N, radius, xmax, ymax, initial_velocity=4):
        
        if (type(radius) == np.ndarray) and len(radius) == N:
            self.radius = radius
            self.rad_max = radius.max()
            self.rad_min = radius.min()
        elif (type(radius)==int) or (type(radius)==float):
            self.radius = np.zeros(N)+radius
            self.rad_max = radius
            self.rad_min = radius

        self._time = 0
        
        ##### Plotting
        self._xmax = xmax + 2*self.rad_max
        self._ymax = ymax + 2*self.rad_max
        self._xmin = 2*self.rad_max
        self._ymin = 2*self.rad_max        
        
        self.fig = plt.figure(figsize=(12,12*(ymax/xmax)))
        self.ax = self.fig.add_axes([0.1,0.1,0.8,0.8],frame_on=False)
        self.ax.set_xticks([]), self.ax.set_yticks([])
        self.ax.set_xlim(self._xmin, self._xmax), self.ax.set_ylim(self._ymin, self._ymax)
        
        self.box=patches.Rectangle((self._xmin,self._ymin),width=xmax,height=ymax,fill=False,lw=3)  ##matplotlib artist for the box
        self.ax.add_patch(self.box)
        
        # Define map grid to locate the particles:
        # each particle will have a label 1 ... N and the entries around its position will carry its label
        # -> accelerates the two-particle collisions, as for every particle we only have to check which other labels are close in the map
        self._map=np.zeros( ( int((xmax+4*self.rad_max)/self.rad_max), int( (xmax+4*self.rad_max)/self.rad_max  ) ) )  
        
        
        
        ########################################################################
		## initialize the positions, more or less uniformly and without overlaps
        ########################################################################
        numx = int(np.sqrt(int(N/xmax*ymax)))+1
        numy = int(N/numx)+1
        stepx = xmax/numx
        stepy = ymax/numy
        assert np.min((stepx,stepy))>self.rad_max
        x = np.linspace(stepx/2,xmax-stepx/2,numx)
        y = np.linspace(stepy/2,ymax-stepy/2,numy)
        positions = np.stack(np.meshgrid(x,y),-1).reshape(-1,2)[0:N] ## mesh, every cell contains one particle
		
        positions[:,0] += np.random.uniform(-(stepx/2-self.rad_max), stepx/2-self.rad_max, N) ## perturb the positions within the cells
        positions[:,1] += np.random.uniform(-(stepy/2-self.rad_max), stepy/2-self.rad_max, N)
		
        positions += 2*self.rad_max ## shift eh positions from [0,xmax] to [2*rad_max,xmax+2*rad_max] for to avoid boundary problems in the map
        
        
        ########################################################################
		# initialize randomly distributed velocities, all with the same magnitude
        ########################################################################
        velangle = np.random.uniform(0,2*np.pi,N)
        velocities = np.stack((np.cos(velangle),np.sin(velangle))).transpose()*initial_velocity
        
        
        ########################################################################
		########## initialize the particles and add their circles to the figure
        ########################################################################
        self.particles=[]
        for i, coord in enumerate(positions):
            self.particles.append(particle(coord[0], coord[1], velocities[i][0], velocities[i][1], radius=self.radius[i]  )   )
            self.ax.add_patch(self.particles[-1]._circle)
            self._map[int(coord[0]/self.rad_max)-1:int(coord[0]/self.rad_max)+2,int(coord[1]/self.rad_max)-1:int(coord[1]/self.rad_max)+2  ]=i+1
            
        mass_norm = self.particles[0]._mass
        infection_seed = self.particles[int(N/2+1)]  ##initialize one sick particle
        infection_seed.get_sick(infection_probability=1)
        mass_sick=infection_seed._mass
        infection_seed._velocity= infection_seed._velocity*mass_norm/mass_sick
        
    def update(self, dt = 0.05):
		
        for i, part in enumerate(self.particles):
            
            part.move(dt)
            
            neighbourhood = self._map[int(part._position[0]/self.rad_max)-1:int(part._position[0]/self.rad_max)+2,int(part._position[1]/self.rad_max)-1:int(part._position[1]/self.rad_max)+2  ]
            neighbourhood[neighbourhood==(i+1)]=0
            aux=np.unique(neighbourhood.reshape(-1))
            collision_inds = aux[aux>0]

            for k in collision_inds:
                part.collision_update(self.particles[int(k)-1])

			
            self._map[int(part._position[0]/self.rad_max)-1:int(part._position[0]/self.rad_max)+2, int(part._position[1]/self.rad_max)-1 : int(part._position[1]/self.rad_max)+2  ] = i+1
            
            part.wall_reflection(self._xmin,self._xmax,self._ymin,self._ymax)
		
        self._time += dt



if __name__ == '__main__':

    ensemble = box_ensemble(N=100, radius=np.random.uniform(0.7,2.5,100), xmax=120, ymax=60, initial_velocity=3)
		
    def animateLive(frame):
        ensemble.update()

    animationLive=FuncAnimation(ensemble.fig, animateLive, frames=900, interval=1)
    animationLive.save('SIR_random_radius.mp4', fps=30, writer='ffmpeg')
    plt.show()
