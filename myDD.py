import time

import dd.cudd as _bdd
from qiskit.converters import circuit_to_dag
import numpy as np

from cir_input.circuit_DG import CreateDGfromQASMfile

r = 2
k = 0
p = 0.001
tmp_noise_num = 0
vrs = []
Fa = []
Fb = []
Fc = []
Fd = []
qd = {}
nd = {}

def Car(A, B, C):
    return (A & B) | (A | B) & C

def xo(A, B):
    return (A & ~B) | (~A & B)

def Sum(A, B, C):
    return xo(xo(A,B), C)

def init_dic(nqubit, nnoise):
    qd = {}
    nd = {}
    for i in range(nqubit):
        qd[i] = i
    for i in range(nnoise):
        nd[i] = [0, 0]
    return qd, nd

def init_BDD(qubit, noise, state):
    global r, k, vrs, Fa, Fb, Fc, Fd, qd, nd
    qd, nd = init_dic(qubit, noise)
    vrs = ['q{i}'.format(i=i) for i in range(qubit+2*noise)]
    bdd.declare(*vrs)
    Fa = [bdd.false for i in range(r)]
    Fb = [bdd.false for i in range(r)]
    Fc = [bdd.false for i in range(r)]
    e = bdd.true
    for i in range(len(state)):
        if(state[i]) == '0':
            e = e & bdd.add_expr(r'~q{i}'.format(i = i))
        else:
            e = e & bdd.add_expr(r'q{i}'.format(i = i))
    Fd = [e]
    Fd.extend([bdd.false for i in range(r-1)])

def apply_H(target):
    global r, k, Fa, Fb, Fc, Fd, qd, nd
    r += 1
    k += 1

    t = qd[target]
    Fa = Fa + [Fa[-1]]
    C = bdd.add_expr(r'q{t}'.format(t = t))
    for i in range(r):
        Gi = bdd.let({r'q{t}'.format(t = t): False}, Fa[i])
        Di = (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fa[i])) | (bdd.add_expr(r'q{t}'.format(t = t)) & ~bdd.let({r'q{t}'.format(t = t): True}, Fa[i]))
        Fa[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

    Fb = Fb + [Fb[-1]]
    C = bdd.add_expr(r'q{t}'.format(t = t))
    for i in range(r):
        Gi = bdd.let({r'q{t}'.format(t = t): False}, Fb[i])
        Di = (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fb[i])) | (bdd.add_expr(r'q{t}'.format(t = t)) & ~bdd.let({r'q{t}'.format(t = t): True}, Fb[i]))
        Fb[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

    Fc = Fc + [Fc[-1]]
    C = bdd.add_expr(r'q{t}'.format(t = t))
    for i in range(r):
        Gi = bdd.let({r'q{t}'.format(t = t): False}, Fc[i])
        Di = (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fc[i])) | (bdd.add_expr(r'q{t}'.format(t = t)) & ~bdd.let({r'q{t}'.format(t = t): True}, Fc[i]))
        Fc[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

    Fd = Fd + [Fd[-1]]
    C = bdd.add_expr(r'q{t}'.format(t = t))
    for i in range(r):
        Gi = bdd.let({r'q{t}'.format(t = t): False}, Fd[i])
        Di = (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fd[i])) | (bdd.add_expr(r'q{t}'.format(t = t)) & ~bdd.let({r'q{t}'.format(t = t): True}, Fd[i]))
        Fd[i] = Sum(Gi, Di, C)
        C = Car(Gi, Di, C)

def apply_X(target):
    global r, k, Fa, Fb, Fc, Fd, qd, nd
    t = qd[target]
    for i in range(r):
        Fa[i] = (bdd.add_expr(r'q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): False}, Fa[i])) | (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fa[i]))
        Fb[i] = (bdd.add_expr(r'q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): False}, Fb[i])) | (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fb[i]))
        Fc[i] = (bdd.add_expr(r'q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): False}, Fc[i])) | (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fc[i]))
        Fd[i] = (bdd.add_expr(r'q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): False}, Fd[i])) | (bdd.add_expr(r'~q{t}'.format(t = t)) & bdd.let({r'q{t}'.format(t = t): True}, Fd[i]))

def apply_CNOT(control, target):
    global r, k, Fa, Fb, Fc, Fd, qd, nd
    c = qd[control]
    t = qd[target]
    for i in range(r):
        Fa[i] = (bdd.add_expr(r'~q{c}'.format(c = c)) & Fa[i]) | (bdd.add_expr(r'q{c} & q{t}'.format(c = c, t = t)) &  bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): False}, Fa[i])) | (bdd.add_expr(r'q{c} & ~q{t}'.format(c = c, t = t))  & bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): True}, Fa[i]))
        Fb[i] = (bdd.add_expr(r'~q{c}'.format(c = c)) & Fb[i]) | (bdd.add_expr(r'q{c} & q{t}'.format(c = c, t = t)) &  bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): False}, Fb[i])) | (bdd.add_expr(r'q{c} & ~q{t}'.format(c = c, t = t))  & bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): True}, Fb[i]))
        Fc[i] = (bdd.add_expr(r'~q{c}'.format(c = c)) & Fc[i]) | (bdd.add_expr(r'q{c} & q{t}'.format(c = c, t = t)) &  bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): False}, Fc[i])) | (bdd.add_expr(r'q{c} & ~q{t}'.format(c = c, t = t))  & bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): True}, Fc[i]))
        Fd[i] = (bdd.add_expr(r'~q{c}'.format(c = c)) & Fd[i]) | (bdd.add_expr(r'q{c} & q{t}'.format(c = c, t = t)) &  bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): False}, Fd[i])) | (bdd.add_expr(r'q{c} & ~q{t}'.format(c = c, t = t))  & bdd.let({r'q{c}'.format(c = c): True, r'q{t}'.format(t = t): True}, Fd[i]))

def apply_SWAP(control, target):
    global r, k, Fa, Fb, Fc, Fd, qd, nd
    apply_CNOT(control, target)
    apply_CNOT(target, control)
    apply_CNOT(control, target)

def apply_noise(target, nqubit):
    global r, k, Fa, Fb, Fc, Fd, qd, nd, tmp_noise_num
    s = nqubit + 2 * tmp_noise_num
    ss = s + 1
    for i in range(r):
        Fa[i] = (bdd.add_expr(r'q{s}'.format(s = s)) & bdd.add_expr(r'q{ss}'.format(ss = ss)) & Fa[i]) | (bdd.add_expr(r'~q{s}'.format(s = s)) & bdd.add_expr(r'~q{ss}'.format(ss = ss)) & Fa[i])
        Fb[i] = (bdd.add_expr(r'q{s}'.format(s = s)) & bdd.add_expr(r'q{ss}'.format(ss = ss)) & Fb[i]) | (bdd.add_expr(r'~q{s}'.format(s = s)) & bdd.add_expr(r'~q{ss}'.format(ss = ss)) & Fb[i])
        Fc[i] = (bdd.add_expr(r'q{s}'.format(s = s)) & bdd.add_expr(r'q{ss}'.format(ss = ss)) & Fc[i]) | (bdd.add_expr(r'~q{s}'.format(s = s)) & bdd.add_expr(r'~q{ss}'.format(ss = ss)) & Fc[i])
        Fd[i] = (bdd.add_expr(r'q{s}'.format(s = s)) & bdd.add_expr(r'q{ss}'.format(ss = ss)) & Fd[i]) | (bdd.add_expr(r'~q{s}'.format(s = s)) & bdd.add_expr(r'~q{ss}'.format(ss = ss)) & Fd[i])

    nd[tmp_noise_num] = [qd[target], s]
    qd[target] = ss

    tmp_noise_num += 1


def apply_cir(dag, nqubit, noise = 0):
    cnt = 0
    for node in dag.topological_op_nodes():
        operating_qubits = [x.index for x in node.qargs]
        if node.name == 'cx':
            apply_CNOT(operating_qubits[0], operating_qubits[1])
        elif node.name =='swap':
            apply_SWAP(operating_qubits[0], operating_qubits[1])
        elif node.name == 'h':
            apply_H(operating_qubits[0])
        elif node.name == 'x':
            apply_X(operating_qubits[0])
        else:
            apply_H(operating_qubits[0])    #todo: SDG
        if cnt < noise:
            apply_noise(operating_qubits[0], nqubit)
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

def to_int_list(l):
    result = []
    for i in l:
        if i == 'FALSE':
            result.append(0)
        else:
            result.append(1)
    return result

def measure(str):
    dic = {}
    qubit = len(str)
    for i in range(qubit):
        dic[r'q{i}'.format(i = i)] = (str[i] == '1')

    a = to_int_list([bdd.to_expr(bdd.let(dic, Fa[j])) for j in range(r)])
    b = to_int_list([bdd.to_expr(bdd.let(dic, Fb[j])) for j in range(r)])
    c = to_int_list([bdd.to_expr(bdd.let(dic, Fc[j])) for j in range(r)])
    d = to_int_list([bdd.to_expr(bdd.let(dic, Fd[j])) for j in range(r)])

    num_a = list_to_num(a)
    num_b = list_to_num(b)
    num_c = list_to_num(c)
    num_d = list_to_num(d)

    coef = ((num_d - num_b) + (num_c - num_a)*1.0j)/(np.sqrt(2))**k

    return np.absolute(coef)

def measure_Ek(str, ek_num, nqubit, nnoise):
    global r, k, Fa, Fb, Fc, Fd, qd, nd, tmp_noise_num
    id = ek_num
    coef = [[[0, 0], [0, 0]] for i in range(nnoise)]
    result = 0
    for j in range(nnoise):
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

    for i in range(4**nnoise):
        dic = {}
        num = i
        tmp_coef = 1.0

        for j in range(nnoise):
            tmp = num % 4
            if tmp == 0:
                dic[r'q{i}'.format(i = nd[j][0])] = False
                dic[r'q{i}'.format(i = nd[j][1])] = False
                tmp_coef *= coef[j][0][0]
            if tmp == 1:
                dic[r'q{i}'.format(i = nd[j][0])] = False
                dic[r'q{i}'.format(i = nd[j][1])] = True
                tmp_coef *= coef[j][0][1]
            if tmp == 2:
                dic[r'q{i}'.format(i = nd[j][0])] = True
                dic[r'q{i}'.format(i = nd[j][1])] = False
                tmp_coef *= coef[j][1][0]
            if tmp == 3:
                dic[r'q{i}'.format(i = nd[j][0])] = True
                dic[r'q{i}'.format(i = nd[j][1])] = True
                tmp_coef *= coef[j][1][1]
            num = int((num-tmp)/4)
        for j in range(nqubit):
            dic[r'q{i}'.format(i = qd[j])] = (str[j] == '1')

        a = to_int_list([bdd.to_expr(bdd.let(dic, Fa[j])) for j in range(r)])
        b = to_int_list([bdd.to_expr(bdd.let(dic, Fb[j])) for j in range(r)])
        c = to_int_list([bdd.to_expr(bdd.let(dic, Fc[j])) for j in range(r)])
        d = to_int_list([bdd.to_expr(bdd.let(dic, Fd[j])) for j in range(r)])

        num_a = list_to_num(a)
        num_b = list_to_num(b)
        num_c = list_to_num(c)
        num_d = list_to_num(d)

        co = ((num_d - num_b) + (num_c - num_a)*1.0j)/(np.sqrt(2))**k
        result += co * tmp_coef

    return result

def noise_measure(str, nqubit, nnoise):
    global r, k, Fa, Fb, Fc, Fd, qd, nd, tmp_noise_num
    result = 0
    for ek in range(4**nnoise):
        result += measure_Ek(str, ek, nqubit, nnoise)
    return result

if __name__ == '__main__':
    #my1()
    path = 'test/'
    file_name = 'cx_30.qasm'
    cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
    #dag_cir = res[0]
    dag = circuit_to_dag(cir)

    # nqubits = cir.num_qubits
    # str = "1"*nqubits
    # bdd = _bdd.BDD()
    # init_BDD(nqubits, 0, "10")
    # #print(bdd.to_expr(Fd[0]))
    # apply_cir(dag, nqubits)
    # # print(bdd.to_expr(Fd[0]))
    # print(measure(str))

    t_start = time.time()

    nqubits = cir.num_qubits
    noise_num = 1
    str = "1"*(nqubits-1)+"0"
    print(nqubits)

    bdd = _bdd.BDD()
    init_BDD(nqubits, noise_num, "0"*nqubits)
    apply_cir(dag, nqubits, noise_num)
    print(noise_measure(str, nqubits, noise_num))

    run_time = time.time() - t_start
    print(run_time)