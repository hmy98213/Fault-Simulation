lambda10 = 1.0866
lambda20 = -0.0866
from scipy import stats, integrate
import math
import numpy as np
import scipy
tmax = 45.575
g = 0.025 * 2 * math.pi
thetaf = math.pi/2
thetai = math.atan(2*g/(0.31))

def faulty_gate_2(noise, tmax, index):
    
    def theta(t, t0):#Fourier approximation of Slepian using 2 elements
        lambda1 = (thetaf-thetai)/t0*lambda10
        lambda2 = (thetaf-thetai)/t0*lambda20
        if( 0 <= t < t0):
            return thetai + ((lambda1+lambda2)*t - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*t/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*t/t0))
        elif (t0 <= t <= 2*t0):
            return (thetaf) - ((lambda1+lambda2)*(t-t0) - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*(t-t0)/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*(t-t0)/t0))   
    def theta_noise(t, t0, lambda10, lambda20): #Fourier approximation of Slepian using 2 elements
        lambda1 = (thetaf-thetai)/t0*lambda10
        lambda2 = (thetaf-thetai)/t0*lambda20
        if( 0 <= t < t0):
            return thetai + ((lambda1+lambda2)*t - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*t/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*t/t0))
        elif (t0 <= t <= 2*t0):
            return (thetaf) - ((lambda1+lambda2)*(t-t0) - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*(t-t0)/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*(t-t0)/t0))   
    def theta_noise1_value(x):
        return math.tan(theta_noise(x, tmax/2, lambda10 * (1 + noise), lambda20)/2)
    
    def theta_noise2_value(x):
        return math.tan(theta_noise(x, tmax/2, lambda10, lambda20 * (1 + noise))/2)
    def theta_value(x):
        return math.tan(theta(x, tmax/2)/2)
    
    sg2 = math.sqrt(2) * g
    varphi1 = scipy.integrate.quad(theta_noise1_value, 0, tmax)[0]
    varphi2 = scipy.integrate.quad(theta_noise2_value, 0, tmax)[0]
    varphi = scipy.integrate.quad(theta_value,0,tmax)[0]
    if(index == 1): 
        return sg2 * (varphi1 - varphi)
    else:
        return sg2 * (varphi2 - varphi)

if __name__ =="__main__":

    X = np.linspace(-0.2, 0.2, 100)
    Y1 = np.zeros(100)
    Y2 = np.zeros(100)
    for index, noise in enumerate(X):
        Y1[index] = faulty_gate_2(noise, 79.925,1)
        Y2[index] = faulty_gate_2(noise, 79.925,2)

    print(Y1)
    print(Y2)