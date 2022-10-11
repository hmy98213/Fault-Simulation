from qiskit import QuantumCircuit
import random

def rand_key(p):
   
    # Variable to store the
    # string
    key1 = ""
 
    # Loop to find the string
    # of desired length
    for i in range(p):
         
        # randint function to generate
        # 0, 1 randomly and converting
        # the result into str
        temp = str(random.randint(0, 1))
 
        # Concatenation the random 0, 1
        # to the final result
        key1 += temp
         
    return(key1)

def gen_bv(qubits, hiddenString):
    
    cir = QuantumCircuit(qubits, qubits)

    for i in range(qubits - 1):
        cir.h(i)

    cir.x(qubits - 1)
    cir.h(qubits - 1)
    hiddenString = list(hiddenString)
    for i in range(len(hiddenString)):
        if hiddenString[i] == "1":
            cir.cx(i, qubits - 1)

    for i in range(qubits):
        cir.h(i)

    return cir.qasm()

if __name__ == "__main__":
    path = "Benchmarks/"
    qubits = 5000
    hs = rand_key(qubits-1)
    print(hs)
    file_name = "bv1_n%d.qasm"%qubits
    with open(path+file_name, 'w') as f:
        f.write(gen_bv(qubits, hs))