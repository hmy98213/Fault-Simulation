from pyeda.inter import *
from qiskit.converters import circuit_to_dag
import numpy as np

from cir_input.circuit_DG import CreateDGfromQASMfile

r = 2
tmp_n = 0
k = 0
p = 0.001
Fa = []
Fb = []
Fc = []
Fd = []

def Car(A, B, C):
    return (A&B)|(A|B)&C

def xo(A, B):
    return (A & ~B)|(~A & B)

def Sum(A, B, C):
    return xo(xo(A,B), C)

def init_BDD(qubit, noise, state):
    q = bddvars('q', qubit+2*noise)
    Fa = [(q[0] & ~q[0]) for i in range(r)]
    Fb = [(q[0] & ~q[0]) for i in range(r)]
    Fc = [(q[0] & ~q[0]) for i in range(r)]
    e = q[0] | ~q[0]
    for i in range(len(state)):
        if(state[i]) == '0':
            e = e & ~q[i]
        else:
            e = e & q[i]
    Fd = [e]
    Fd.extend([(q[0] & ~q[0]) for i in range(r-1)])
    return q, Fa, Fb, Fc, Fd

def reorder(f, qubit, t, i):
    result = f.compose({q[t]: q[qubit+2*i], q[qubit+2*i]: q[t]})
    return result

def apply_H(t, q):
    global r, k, Fa, Fb, Fc, Fd
    r += 1
    k += 1

    Fa = Fa + [Fa[-1]]
    C = q[t]
    for i in range(r):
        Gi = Fa[i].restrict({q[t]: 0})
        Di = (~q[t] & Fa[i].restrict({q[t]: 1})) | (q[t] & ~Fa[i].restrict({q[t]: 1}))
        Fa[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

    Fb = Fb + [Fb[-1]]
    C = q[t]
    for i in range(r):
        Gi = Fb[i].restrict({q[t]: 0})
        Di = (~q[t] & Fb[i].restrict({q[t]: 1})) | (q[t] & ~Fb[i].restrict({q[t]: 1}))
        Fb[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

    Fc = Fc + [Fc[-1]]
    C = q[t]
    for i in range(r):
        Gi = Fc[i].restrict({q[t]: 0})
        Di = (~q[t] & Fc[i].restrict({q[t]: 1})) | (q[t] & ~Fc[i].restrict({q[t]: 1}))
        Fc[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

    Fd = Fd + [Fd[-1]]
    C = q[t]
    for i in range(r):
        Gi = Fd[i].restrict({q[t]: 0})
        Di = (~q[t] & Fd[i].restrict({q[t]: 1})) | (q[t] & ~Fd[i])
        Fd[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

def apply_X(t, q):
    global r, k, Fa, Fb, Fc, Fd
    for i in range(r):
        Fa[i] = (q[t] & Fa[i].restrict({q[t]: 0})) | (~q[t] & Fa[i].restrict({q[t]: 1}))
        Fb[i] = (q[t] & Fb[i].restrict({q[t]: 0})) | (~q[t] & Fb[i].restrict({q[t]: 1}))
        Fc[i] = (q[t] & Fc[i].restrict({q[t]: 0})) | (~q[t] & Fc[i].restrict({q[t]: 1}))
        Fd[i] = (q[t] & Fd[i].restrict({q[t]: 0})) | (~q[t] & Fd[i].restrict({q[t]: 1}))

def apply_CNOT(c, t, q):
    global r, k, Fa, Fb, Fc, Fd
    for i in range(r):
        Fa[i] = (~q[c] & Fa[i]) | (q[c] & q[t] & Fa[i].restrict({q[c]: 1, q[t]: 0})) | (q[c] & ~q[t] & Fa[i].restrict({q[c]: 1, q[t]: 1}))
        Fb[i] = (~q[c] & Fb[i]) | (q[c] & q[t] & Fb[i].restrict({q[c]: 1, q[t]: 0})) | (q[c] & ~q[t] & Fb[i].restrict({q[c]: 1, q[t]: 1}))
        Fc[i] = (~q[c] & Fc[i]) | (q[c] & q[t] & Fc[i].restrict({q[c]: 1, q[t]: 0})) | (q[c] & ~q[t] & Fc[i].restrict({q[c]: 1, q[t]: 1}))
        Fd[i] = (~q[c] & Fd[i]) | (q[c] & q[t] & Fd[i].restrict({q[c]: 1, q[t]: 0})) | (q[c] & ~q[t] & Fd[i].restrict({q[c]: 1, q[t]: 1}))

def apply_noise(t, qubit ,q):
    global r, k, Fa, Fb, Fc, Fd, tmp_n
    s = qubit + 2*tmp_n
    ss = s+1
    for i in range(r):
        Fa[i] = (q[s] & q[ss] & Fa[i]) | (~q[s] & ~q[ss] & Fa[i])
        Fb[i] = (q[s] & q[ss] & Fb[i]) | (~q[s] & ~q[ss] & Fb[i])
        Fc[i] = (q[s] & q[ss] & Fc[i]) | (~q[s] & ~q[ss] & Fc[i])
        Fd[i] = (q[s] & q[ss] & Fd[i]) | (~q[s] & ~q[ss] & Fd[i])
    for i in range(r):
        Fa[i] = reorder(Fa[i], qubit, t, tmp_n)
        Fb[i] = reorder(Fb[i], qubit, t, tmp_n)
        Fc[i] = reorder(Fc[i], qubit, t, tmp_n)
        Fd[i] = reorder(Fd[i], qubit, t, tmp_n)



def apply_cir(dag, q, qubit, noise = 0):
    cnt = 0
    for node in dag.topological_op_nodes():
        operating_qubits = [x.index for x in node.qargs]
        if node.name == 'cx':
            apply_CNOT(operating_qubits[0], operating_qubits[1], q)
        elif node.name == 'h':
            apply_H(operating_qubits[0], q)
        elif node.name == 'x':
            apply_X(operating_qubits[0], q)
        if cnt < noise:
            apply_noise(operating_qubits[0], qubit, q)
            cnt += 1

def list_to_num(l):
    num = 0
    id = 0
    for i in l[:-1]:
        num += i*2**id
        id += 1
    if l[-1] == 0:
        return num
    else:
        return num - 2**id


def measure(str, q):
    dic = {}
    qubit = len(str)
    for i in range(qubit):
        dic[q[i]] = str[i]
    a = [int(Fa[j].restrict(dic)) for j in range(r)]
    b = [int(Fb[j].restrict(dic)) for j in range(r)]
    c = [int(Fc[j].restrict(dic)) for j in range(r)]
    d = [int(Fd[j].restrict(dic)) for j in range(r)]

    num_a = list_to_num(a)
    num_b = list_to_num(b)
    num_c = list_to_num(c)
    num_d = list_to_num(d)

    coef = ((num_d - num_b) + (num_c - num_a)*1.0j)/(np.sqrt(2))**k

    return np.absolute(coef)

def measure_Ek(str, ek_num, qubit, noise, q):
    global p, k
    id = ek_num
    coef = [[[0, 0], [0, 0]] for i in range(noise)]
    result = 0
    for j in range(noise):
        tmp = id % 4
        if tmp == 0:
            coef[j][0][0] = 1 - p
            coef[j][1][1] = 1 - p
        elif tmp == 1:
            coef[j][0][1] = p/3
            coef[j][1][0] = p/3
        elif tmp == 2:
            coef[j][0][1] = -1.0j/3 * p
            coef[j][1][0] = 1.0j/3 * p
        elif tmp ==3:
            coef[j][0][0] = 1/3 * p
            coef[j][1][1] = -1/3 * p
        id = (id-tmp)/4

    for i in range(4**noise):
        dic = {}
        num = i
        tmp_coef = 1.0

        for j in range(noise):
            tmp = num % 4
            if tmp == 0:
                dic[q[qubit+2*j]] = 0
                dic[q[qubit+2*j+1]] = 0
                tmp_coef *= coef[j][0][0]
            if tmp == 1:
                dic[q[qubit+2*j]] = 0
                dic[q[qubit+2*j+1]] = 1
                tmp_coef *= coef[j][0][1]
            if tmp == 2:
                dic[q[qubit+2*j]] = 1
                dic[q[qubit+2*j+1]] = 0
                tmp_coef *= coef[j][1][0]
            if tmp == 3:
                dic[q[qubit+2*j]] = 1
                dic[q[qubit+2*j+1]] = 1
                tmp_coef *= coef[j][1][1]
            num = int((num-tmp)/4)
        for j in range(qubit):
            dic[q[j]] = str[j]

        a = [int(Fa[j].restrict(dic)) for j in range(r)]
        b = [int(Fb[j].restrict(dic)) for j in range(r)]
        c = [int(Fc[j].restrict(dic)) for j in range(r)]
        d = [int(Fd[j].restrict(dic)) for j in range(r)]
        num_a = list_to_num(a)
        num_b = list_to_num(b)
        num_c = list_to_num(c)
        num_d = list_to_num(d)
        co = ((num_d - num_b) + (num_c - num_a)*1.0j)/(np.sqrt(2))**k
        result += co * tmp_coef

    return result


def noise_measure(str, qubit, noise, q):
    result = 0
    for k in range(4**noise):
        result += measure_Ek(str, k, qubit, noise, q)
    return result


def my1():
    global r, k, Fa, Fb, Fc, Fd, q
    path = 'Benchmarks/'
    file_name = 'h.qasm'  # h.qasm
    cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
    dag = circuit_to_dag(cir)

    nqubits = cir.num_qubits
    str = "0"*nqubits
    q, Fa, Fb, Fc, Fd = init_BDD(nqubits, 2, "00")
    apply_cir(dag, q, nqubits)
    print(measure("00", q))
    print(measure("01", q))
    print(measure("10", q))
    print(measure("11", q))

if __name__ == '__main__':
    #my1()
    path = 'Benchmarks/'
    file_name = 'bv_n15.qasm'
    cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
    #dag_cir = res[0]
    dag = circuit_to_dag(cir)

    nqubits = cir.num_qubits
    str = "1"*nqubits
    q, Fa, Fb, Fc, Fd = init_BDD(nqubits, 0, "1"*nqubits)
    apply_cir(dag, q, nqubits)
    print(measure(str, q))

    # nqubits = cir.num_qubits
    # noise_num = 1
    # str = "0"*nqubits
    # q, Fa, Fb, Fc, Fd = init_BDD(nqubits, noise_num, "1"*nqubits)
    # apply_cir(dag, q, nqubits, noise_num)
    # print(noise_measure(str, nqubits, noise_num, q))



