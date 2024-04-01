import numpy as np

def calculate_precision(p, n):
    return (1+8*p)**n-1-n*4*p*(1+4*p)**(n-1)

def sample_number(t):
    return np.log(100)/(2*t**2)

if __name__ == '__main__':
    for n in range(10 , 41):
        print("(%d, %d)"%(n, 6*n+2))