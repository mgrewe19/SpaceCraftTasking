import gym
from gym import spaces
import numpy as np
import SpaceCraftTaskingEnviornment #Connects the enviornment file (class)

class Simulator():
    def __init__(self, IC = None):
        if IC is not None:
            self.state = IC
        else:
            self.new_IC()
        #print(self.state)
        self.dt = 10 #Seconds
        #self.N = 0.0011 #Mean motion at ~500 km
        self.mu = 1.327124e20 #(m^3/s^2)mu of the sun
        self.radius_bennu = 282.5 #Largest radius of Bennu
        massBennu = 7.329e10 #kg this is the mass of Bennu
        G = 6.67408e-11 #(m^3/(kg*s^2)) Gravitational constant
        self.muBennu = (massBennu * G) #(m^3/s^2) mu of Bennu
        self.StepSize = 30 #This initalizes the step size. (In seconds)
        self.Current_Reward = 0 #Initializes the current reward to zero
        self.reflectivity = 0.4 #Sets the value of the reflectivity of the spacecraft
        self.SRP = 4.56e-6 #Sets the SRP of the space craft
        self.massSC = 50 #Mass of the spacecraft (kg)
        self.areaSC = 0.79 #Area of the spacecrafts surface (m^2)
        self.au = 1.4959787e11 #Value of 1 AU (m)
    def new_IC(self):
        self.state = np.array([])
    def propogate_state(self, x_k):
        """Integrates the spacecraft dynamics forward by one time step using the RK4 method """
        #print(x_k)

        new_state = np.zeros((12,1))
        
        #print(self.state)
        k1 = self.dt*self.equations_of_motion(x_k)
        #print("k1: ", k1)
        k2 = self.dt*self.equations_of_motion(x_k + 0.5*k1[:,0])
        #print("k2: ", k2)
        k3 = self.dt*self.equations_of_motion(x_k + 0.5*k2[:,0])
        #print("k3: ", k3)
        k4 = self.dt*self.equations_of_motion(x_k + k3[:,0])
        #print("k4: ", k4)
        self.state += (1.0/6.0)*(k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) #Divide by 6 becuase that is 
            #How many terms you are adding together

        #print(self.state)
        return self.state

    def equations_of_motion(self, x_k):
        #Im still not sure what x_k is supposed to be doing here
            #Is x_k the equation of state? 1-3 are position, 4-6 are velocity
        """Returns the x_dot vector"""
        #Initalizes x
        x_dot = np.zeros((12,1))
        #print(x_k)

    #The following information is for Bennu
        #Velocity
        x_dot[0] = x_k[3]
        x_dot[1] = x_k[4]
        x_dot[2] = x_k[5]

        #Acceleration
        d = np.linalg.norm(x_k[0:3])
        x_dot[3] = -self.mu*x_k[0]/(d**3)
        x_dot[4] = -self.mu*x_k[1]/(d**3)
        x_dot[5] = -self.mu*x_k[2]/(d**3)

    #The following information is for the spacecraft orbiting around Bennu
        #Velocity
        x_dot[6] = x_k[9]
        x_dot[7] = x_k[10]
        x_dot[8] = x_k[11] 

        #Acceleration (These equations are not correct yet)
        Identiy = np.array([[1, 0, 0], [0, 1 , 0], [0, 0 , 1]])
        r2 = np.array([x_k[6:9]]).T #Vector from the spacecraft to the asteroid
        Norm2 = np.linalg.norm(r2) #Trying to take the norm of the diff of two vectors 
            #(Distance from spacecraft to Bennu)
        #Ra = self.muBennu/Norm2 #This calculates the gravitational potential of the asteroid
        #RdotR = (r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]) #This is the calulation of r2 dot r2
        d_hat = - np.array(x_k[0:3]/d) #This calulates the unit vector of d to the sun
        #print(d_hat)
        #RdotD = (r2[0]*d_hat[0] + r2[1]*d_hat[1] + r2[2]*d_hat[2])
        #Rs = (-self.mu/(2 * r**3)) * (RdotR - 3* (RdotD**2))

        Calc1 = (3*np.matmul(d_hat, np.matrix.transpose(d_hat))) #First step of calculation
            #3*D*D^T (3 times vector to the sun times transpose of vector to the sun)

        Calc2 = np.subtract(Calc1, Identiy) #Second step of calculation
            #Calc1 - I[3x3]

        Calc3 = np.matmul(Calc2, r2) #Does the final inbetween step of multiplying Calc2 by the distance of 
            #the spacecraft to the asteroid

        x_dot[9:12] = -self.muBennu*(np.array([x_k[6:9]]).T)/(Norm2**3) + (self.mu/d**3)*(Calc3)

        #Calculation for the acceleration due to solar radiation

        #calc 4
        Calc4 = self.SRP*(1 + self.reflectivity) #P_o * (1 + rho)

        #Calc 5
        Calc5 = Calc4 * self.au**2 * self.areaSC / self.massSC #Calc4 * AU^2 * A_SC / M_sc

        #r - d (Note that d is -x_k[0:3])
        rd = x_k[6:9] + x_k[0:3]

        normRD = np.linalg.norm(rd) #|r-d|

        #Calc6
        Calc6 = Calc5 * rd / (normRD)**3 # Calc5 * (r-d)/|r-d|^3

        x_dot[9:12] = (np.array(x_dot[9:12]).T + np.array(Calc6).T).T 
            #Combines the acceleration from the gravity of bennu and solar radiation

        return x_dot

    def get_reward(self):
        distance_from_center = np.linalg.norm(self.state[6:9])
        if (distance_from_center <= self.radius_bennu):
            return -10
        else:
            return 0

    def step(self):
        #print(self.state)
        for i in range(int(self.StepSize/self.dt)):
            self.propogate_state(self.state)
            self.Current_Reward = self.Current_Reward + self.get_reward()
        return self.state