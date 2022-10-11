import class_DD
import multiprocessing
import os
import time
import psutil
import dd.cudd as _bdd
from qiskit.converters import circuit_to_dag
import numpy as np

from cir_input.circuit_DG import CreateDGfromQASMfile

import signal
from contextlib import contextmanager

from cir_input.circuit_process import get_gates_number
from multiprocessing import Pool, Process, Queue
from joblib import Parallel, delayed

class TimeoutException(Exception): pass



@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class cirDD:
    r = 2
    k = 0
    p = 0.001
    tmp_noise_num = 0
    max_bdd = 0
    vrs = []
    Fa = []
    Fb = []
    Fc = []
    Fd = []
    qd = {}
    nd = {}

    def init_dic(self, nqubit, nnoise):
        self.qd = {}
        self.nd = {}
        for i in range(nqubit):
            self.qd[i] = i
        for i in range(nnoise):
            self.nd[i] = [0, 0]

    def __init__(self, nqubit, nnoise, state):
        self.init_dic(nqubit, nnoise)
        self.vrs = ['q{i}'.format(i=i) for i in range(nqubit + 2 * nnoise)]
        bdd.declare(*self.vrs)
        self.Fa = [bdd.false for i in range(self.r)]
        self.Fb = [bdd.false for i in range(self.r)]
        self.Fc = [bdd.false for i in range(self.r)]
        e = bdd.true
        for i in range(len(state)):
            if (state[i]) == '0':
                e = e & bdd.add_expr(r'~q{i}'.format(i=i))
            else:
                e = e & bdd.add_expr(r'q{i}'.format(i=i))
        self.Fd = [e]
        self.Fd.extend([bdd.false for i in range(self.r - 1)])

    def check_zero(self):
        for i in range(self.r-2, -1, -1):
            if(self.Fa[i]==self.Fa[self.r-1] and self.Fb[i]==self.Fb[self.r-1] and self.Fc[i]==self.Fc[self.r-1] and self.Fd[i]==self.Fd[self.r-1]):
                self.Fa.pop(i)
                self.Fb.pop(i)
                self.Fc.pop(i)
                self.Fd.pop(i)
                self.r -= 1
            else:
                break
        while(self.r>2 and self.Fa[0]==bdd.false and self.Fb[0]==bdd.false and self.Fc[0]==bdd.false and self.Fd[0]==bdd.false):
            self.Fa.pop(0)
            self.Fb.pop(0)
            self.Fc.pop(0)
            self.Fd.pop(0)
            self.r -= 1
            self.k -= 2

    def bound_bdd(self):
        if(self.max_bdd == 0):
            return
        while(self.r > self.max_bdd):
            self.Fa.pop(0)
            self.Fb.pop(0)
            self.Fc.pop(0)
            self.Fd.pop(0)
            self.r -= 1
            self.k -= 2


    def apply_H(self, target):
        self.r += 1
        self.k += 1
        t = self.qd[target]

        self.Fa = self.Fa + [self.Fa[-1]]
        self.Fb = self.Fb + [self.Fb[-1]]
        self.Fc = self.Fc + [self.Fc[-1]]
        self.Fd = self.Fd + [self.Fd[-1]]

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = bdd.let({r'q{t}'.format(t=t): False}, self.Fa[i])
            Di = (bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fa[i])) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~bdd.let({r'q{t}'.format(t=t): True}, self.Fa[i]))
            self.Fa[i] = Sum(Gi, Di, C)
            C = Car(Gi, Di, C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = bdd.let({r'q{t}'.format(t=t): False}, self.Fb[i])
            Di = (bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fb[i])) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~bdd.let({r'q{t}'.format(t=t): True}, self.Fb[i]))
            self.Fb[i] = Sum(Gi, Di, C)
            C = Car(Gi, Di, C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = bdd.let({r'q{t}'.format(t=t): False}, self.Fc[i])
            Di = (bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fc[i])) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~bdd.let({r'q{t}'.format(t=t): True}, self.Fc[i]))
            self.Fc[i] = Sum(Gi, Di, C)
            C = Car(Gi, Di, C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        
        for i in range(self.r):
            tmp_t = time.time()
            Gi = bdd.let({r'q{t}'.format(t=t): False}, self.Fd[i])
            # print("compute Gi time" + str(time.time()-tmp_t))
            tmp_t = time.time()
            # Di = (bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fd[i])) | (
            #             bdd.add_expr(r'q{t}'.format(t=t)) & ~bdd.let({r'q{t}'.format(t=t): True}, self.Fd[i]))
            Di = (bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fd[i])) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~ self.Fd[i])
            # print("compute Di time" + str(time.time()-tmp_t))
            tmp_t = time.time()
            self.Fd[i] = Sum(Gi, Di, C)
            # print("compute Sum time" + str(time.time()-tmp_t))
            tmp_t = time.time()
            C = Car(Gi, Di, C)
            # print("compute Carry time" + str(time.time()-tmp_t))


    def apply_X(self, target):
        t = self.qd[target]
        for i in range(self.r):
            self.Fa[i] = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fa[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fa[i]))
            self.Fb[i] = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fb[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fb[i]))
            self.Fc[i] = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fc[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fc[i]))
            self.Fd[i] = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fd[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fd[i]))

    def apply_Y(self, target):
        self.r += 1
        self.k += 1
        t = self.qd[target]

        self.Fa = self.Fa + [self.Fa[-1]]
        self.Fb = self.Fb + [self.Fb[-1]]
        self.Fc = self.Fc + [self.Fc[-1]]
        self.Fd = self.Fd + [self.Fd[-1]]

        C = bdd.add_expr(r'~q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fc[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fc[i]))
            Di = (bdd.add_expr(r'q{t}'.format(t=t)) & Gi) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & ~Gi)
            self.Fa[i] = Sum(Di, bdd.false, C)
            C = Car(Di, bdd.false, C)

        C = bdd.add_expr(r'~q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fd[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fd[i]))
            Di = (bdd.add_expr(r'q{t}'.format(t=t)) & Gi) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & ~Gi)
            self.Fa[i] = Sum(Di, bdd.false, C)
            C = Car(Di, bdd.false, C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fa[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fa[i]))
            Di = (bdd.add_expr(r'q{t}'.format(t=t)) & ~Gi) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & Gi)
            self.Fa[i] = Sum(Di, bdd.false, C)
            C = Car(Di, bdd.false, C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): False}, self.Fb[i])) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & bdd.let({r'q{t}'.format(t=t): True}, self.Fb[i]))
            Di = (bdd.add_expr(r'q{t}'.format(t=t)) & ~Gi) | (
                        bdd.add_expr(r'~q{t}'.format(t=t)) & Gi)
            self.Fa[i] = Sum(Di, bdd.false, C)
            C = Car(Di, bdd.false, C)

    def apply_Z(self, target):
        self.r += 1
        self.k += 1
        t = self.qd[target]

        self.Fa = self.Fa + [self.Fa[-1]]
        self.Fb = self.Fb + [self.Fb[-1]]
        self.Fc = self.Fc + [self.Fc[-1]]
        self.Fd = self.Fd + [self.Fd[-1]]

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fa[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fa[i])
            self.Fa[i] = Sum(Gi, bdd.false, C)
            C = Car(Gi, bdd.false,C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fb[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fb[i])
            self.Fb[i] = Sum(Gi, bdd.false,C)
            C = Car(Gi, bdd.false,C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fc[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fc[i])
            self.Fc[i] = Sum(Gi, bdd.false,C)
            C = Car(Gi, bdd.false,C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fd[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fd[i])
            self.Fd[i] = Sum(Gi, bdd.false,C)
            C = Car(Gi, bdd.false,C)

    def apply_S(self, target):
        self.r += 1
        self.k += 1
        t = self.qd[target]

        self.Fa = self.Fa + [self.Fa[-1]]
        self.Fb = self.Fb + [self.Fb[-1]]
        self.Fc = self.Fc + [self.Fc[-1]]
        self.Fd = self.Fd + [self.Fd[-1]]

        for i in range(self.r):
            self.Fa[i] = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fa[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & self.Fc[i])

        for i in range(self.r):
            self.Fb[i] = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fb[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & self.Fd[i])

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fc[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fa[i])
            self.Fc[i] = Sum(Gi, bdd.false,C)
            C = Car(Gi, bdd.false,C)

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fd[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fb[i])
            self.Fd[i] = Sum(Gi, bdd.false,C)
            C = Car(Gi, bdd.false,C)

    def apply_T(self, target):
        self.r += 1
        self.k += 1
        t = self.qd[target]

        self.Fa = self.Fa + [self.Fa[-1]]
        self.Fb = self.Fb + [self.Fb[-1]]
        self.Fc = self.Fc + [self.Fc[-1]]
        self.Fd = self.Fd + [self.Fd[-1]]

        for i in range(self.r):
            self.Fa[i] = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fa[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & self.Fb[i])

        for i in range(self.r):
            self.Fb[i] = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fb[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & self.Fc[i])

        for i in range(self.r):
            self.Fc[i] = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fc[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & self.Fd[i])

        C = bdd.add_expr(r'q{t}'.format(t=t))
        for i in range(self.r):
            Gi = (bdd.add_expr(r'~q{t}'.format(t=t)) & self.Fd[i]) | (
                        bdd.add_expr(r'q{t}'.format(t=t)) & ~self.Fa[i])
            self.Fd[i] = Sum(Gi, bdd.false,C)
            C = Car(Gi, bdd.false,C)


    def apply_CNOT(self, control, target):
        c = self.qd[control]
        t = self.qd[target]
        for i in range(self.r):
            self.Fa[i] = (bdd.add_expr(r'~q{c}'.format(c=c)) & self.Fa[i]) | (
                        bdd.add_expr(r'q{c} & q{t}'.format(c=c, t=t)) & bdd.let(
                    {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): False}, self.Fa[i])) | (
                                bdd.add_expr(r'q{c} & ~q{t}'.format(c=c, t=t)) & bdd.let(
                            {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): True}, self.Fa[i]))
            self.Fb[i] = (bdd.add_expr(r'~q{c}'.format(c=c)) & self.Fb[i]) | (
                        bdd.add_expr(r'q{c} & q{t}'.format(c=c, t=t)) & bdd.let(
                    {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): False}, self.Fb[i])) | (
                                bdd.add_expr(r'q{c} & ~q{t}'.format(c=c, t=t)) & bdd.let(
                            {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): True}, self.Fb[i]))
            self.Fc[i] = (bdd.add_expr(r'~q{c}'.format(c=c)) & self.Fc[i]) | (
                        bdd.add_expr(r'q{c} & q{t}'.format(c=c, t=t)) & bdd.let(
                    {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): False}, self.Fc[i])) | (
                                bdd.add_expr(r'q{c} & ~q{t}'.format(c=c, t=t)) & bdd.let(
                            {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): True}, self.Fc[i]))
            self.Fd[i] = (bdd.add_expr(r'~q{c}'.format(c=c)) & self.Fd[i]) | (
                        bdd.add_expr(r'q{c} & q{t}'.format(c=c, t=t)) & bdd.let(
                    {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): False}, self.Fd[i])) | (
                                bdd.add_expr(r'q{c} & ~q{t}'.format(c=c, t=t)) & bdd.let(
                            {r'q{c}'.format(c=c): True, r'q{t}'.format(t=t): True}, self.Fd[i]))

    def apply_SWAP(self, control, target):
        self.apply_CNOT(control, target)
        self.apply_CNOT(target, control)
        self.apply_CNOT(control, target)

    def apply_noise(self, target, nqubit):
        s = nqubit + 2 * self.tmp_noise_num
        ss = s + 1
        for i in range(self.r):
            self.Fa[i] = (bdd.add_expr(r'q{s}'.format(s=s)) & bdd.add_expr(r'q{ss}'.format(ss=ss)) & self.Fa[i]) | (
                        bdd.add_expr(r'~q{s}'.format(s=s)) & bdd.add_expr(r'~q{ss}'.format(ss=ss)) & self.Fa[i])
            self.Fb[i] = (bdd.add_expr(r'q{s}'.format(s=s)) & bdd.add_expr(r'q{ss}'.format(ss=ss)) & self.Fb[i]) | (
                        bdd.add_expr(r'~q{s}'.format(s=s)) & bdd.add_expr(r'~q{ss}'.format(ss=ss)) & self.Fb[i])
            self.Fc[i] = (bdd.add_expr(r'q{s}'.format(s=s)) & bdd.add_expr(r'q{ss}'.format(ss=ss)) & self.Fc[i]) | (
                        bdd.add_expr(r'~q{s}'.format(s=s)) & bdd.add_expr(r'~q{ss}'.format(ss=ss)) & self.Fc[i])
            self.Fd[i] = (bdd.add_expr(r'q{s}'.format(s=s)) & bdd.add_expr(r'q{ss}'.format(ss=ss)) & self.Fd[i]) | (
                        bdd.add_expr(r'~q{s}'.format(s=s)) & bdd.add_expr(r'~q{ss}'.format(ss=ss)) & self.Fd[i])

        self.nd[self.tmp_noise_num] = [self.qd[target], s]
        self.qd[target] = ss

        self.tmp_noise_num += 1

    def apply_cir(self, dag, nqubit, noise=0):
        cnt = 0
        gate_cnt = 0
        for node in dag.topological_op_nodes():
            operating_qubits = [x.index for x in node.qargs]
            if node.name == 'cx':
                self.apply_CNOT(operating_qubits[0], operating_qubits[1])
            elif node.name == 'swap':
                self.apply_SWAP(operating_qubits[0], operating_qubits[1])
            elif node.name == 'h':
                self.apply_H(operating_qubits[0])
            elif node.name == 'x':
                self.apply_X(operating_qubits[0])
            elif node.name == 'y':
                self.apply_Y(operating_qubits[0])
            elif node.name == 'z':
                self.apply_Z(operating_qubits[0])
            elif node.name == 't':
                self.apply_T(operating_qubits[0])
            elif node.name == 's':
                self.apply_S(operating_qubits[0])
            elif node.name == 'sdg':
                self.apply_Z(operating_qubits[0])
                self.apply_S(operating_qubits[0])
            else:
                raise Exception("No this kind of gate!")
            gate_cnt += 1
            print(node.name+" "+str(gate_cnt)+" "+str(len(self.Fd)))
            if cnt < noise:
                self.apply_noise(operating_qubits[0], nqubit)
                cnt += 1
            self.check_zero()
            self.bound_bdd()


    def apply_bv_cir(self, dag, nqubit, noise=0):
        cnt = 0
        # nqubit = 10000
        for i in range(nqubit-1):
            self.apply_H(i)
            print("h"+str(i)+" "+str(len(self.Fd)))
            self.check_zero()
            self.bound_bdd()
            if cnt < noise:
                self.apply_noise(i, nqubit)
                cnt += 1
        self.apply_X(nqubit-1)
        self.apply_H(nqubit-1)
        for i in range(nqubit-1):
            self.apply_CNOT(i, nqubit-1)
            self.check_zero()
            self.bound_bdd()
            print("cx"+str(i)+" "+str(len(self.Fd)))
            print("h"+str(i)+" "+str(len(self.Fd)))
            self.apply_H(i)
            self.check_zero()
            self.bound_bdd()            


    def measure(self, str):
        dic = {}
        qubit = len(str)
        for i in range(qubit):
            dic[r'q{i}'.format(i=i)] = (str[i] == '1')

        a = to_int_list([bdd.to_expr(bdd.let(dic, self.Fa[j])) for j in range(self.r)])
        b = to_int_list([bdd.to_expr(bdd.let(dic, self.Fb[j])) for j in range(self.r)])
        c = to_int_list([bdd.to_expr(bdd.let(dic, self.Fc[j])) for j in range(self.r)])
        d = to_int_list([bdd.to_expr(bdd.let(dic, self.Fd[j])) for j in range(self.r)])

        num_a = list_to_num(a)
        num_b = list_to_num(b)
        num_c = list_to_num(c)
        num_d = list_to_num(d)

        coef = ((num_d - num_b) + (num_c - num_a) * 1.0j) / (np.sqrt(2)) ** self.k

        return np.absolute(coef)

    def fast_noise_measure(self, str, nqubit, nnoise):
        coef = {}
        result = 0
        for j in range(nnoise):
            coef[j] = {}
            coef[j][0] = [[1 - self.p, 0], [0, 1 - self.p]]
            coef[j][1] = [[0, self.p/3], [self.p/3, 0]]
            coef[j][2] = [[0, -1.0j*self.p/3], [1.0j*self.p/3, 0]]
            coef[j][3] = [[1/3*self.p, 0], [0, -1/3*self.p]]

        for i in range(4 ** nnoise):
            dic = {}
            num = i
            tmp_coef = 1.0

            for j in range(nnoise):
                tmp = num % 4
                if tmp == 0:
                    dic[r'q{i}'.format(i=self.nd[j][0])] = False
                    dic[r'q{i}'.format(i=self.nd[j][1])] = False
                    tmp_coef *= coef[j][0][0][0] + coef[j][1][0][0]+coef[j][2][0][0]+coef[j][3][0][0]
                if tmp == 1:
                    dic[r'q{i}'.format(i=self.nd[j][0])] = False
                    dic[r'q{i}'.format(i=self.nd[j][1])] = True
                    tmp_coef *= coef[j][0][0][1] + coef[j][1][0][1]+coef[j][2][0][1]+coef[j][3][0][1]
                if tmp == 2:
                    dic[r'q{i}'.format(i=self.nd[j][0])] = True
                    dic[r'q{i}'.format(i=self.nd[j][1])] = False
                    tmp_coef *= coef[j][0][1][0] + coef[j][1][1][0]+coef[j][2][1][0]+coef[j][3][1][0]
                if tmp == 3:
                    dic[r'q{i}'.format(i=self.nd[j][0])] = True
                    dic[r'q{i}'.format(i=self.nd[j][1])] = True
                    tmp_coef *= coef[j][0][1][1] + coef[j][1][1][1]+coef[j][2][1][1]+coef[j][3][1][1]
                num = int((num - tmp) / 4)
            for j in range(nqubit):
                dic[r'q{i}'.format(i=self.qd[j])] = (str[j] == '1')

            a = to_int_list([bdd.to_expr(bdd.let(dic, self.Fa[j])) for j in range(self.r)])
            b = to_int_list([bdd.to_expr(bdd.let(dic, self.Fb[j])) for j in range(self.r)])
            c = to_int_list([bdd.to_expr(bdd.let(dic, self.Fc[j])) for j in range(self.r)])
            d = to_int_list([bdd.to_expr(bdd.let(dic, self.Fd[j])) for j in range(self.r)])

            num_a = list_to_num(a)
            num_b = list_to_num(b)
            num_c = list_to_num(c)
            num_d = list_to_num(d)

            co = ((num_d - num_b) + (num_c - num_a) * 1.0j) / (np.sqrt(2)) ** self.k
            result += co * tmp_coef

        return result

    # Calculate each F(\psi_e, J)
    def noise_assign(self, str, idj, nqubit, nnoise, result_dict):
        print(psutil.Process().cpu_num())
        dic = {}
        num = idj
        for j in range(nnoise):
            tmp = num % 4
            if tmp == 0:
                dic[r'q{i}'.format(i=self.nd[j][0])] = False
                dic[r'q{i}'.format(i=self.nd[j][1])] = False
            if tmp == 1:
                dic[r'q{i}'.format(i=self.nd[j][0])] = False
                dic[r'q{i}'.format(i=self.nd[j][1])] = True
            if tmp == 2:
                dic[r'q{i}'.format(i=self.nd[j][0])] = True
                dic[r'q{i}'.format(i=self.nd[j][1])] = False
            if tmp == 3:
                dic[r'q{i}'.format(i=self.nd[j][0])] = True
                dic[r'q{i}'.format(i=self.nd[j][1])] = True
            num = int((num - tmp) / 4)
        for j in range(nqubit):
            dic[r'q{i}'.format(i=self.qd[j])] = (str[j] == '1')

        a = to_int_list([bdd.to_expr(bdd.let(dic, self.Fa[j])) for j in range(self.r)])
        b = to_int_list([bdd.to_expr(bdd.let(dic, self.Fb[j])) for j in range(self.r)])
        c = to_int_list([bdd.to_expr(bdd.let(dic, self.Fc[j])) for j in range(self.r)])
        d = to_int_list([bdd.to_expr(bdd.let(dic, self.Fd[j])) for j in range(self.r)])

        num_a = list_to_num(a)
        num_b = list_to_num(b)
        num_c = list_to_num(c)
        num_d = list_to_num(d)

        result = ((num_d - num_b) + (num_c - num_a) * 1.0j) / (np.sqrt(2)) ** self.k
        result_dict[idj] = result
        #q.put((idj, result))
        #return result


    def class_para_test(self, i):
        return i*i

    def noise_measure(self, str, nqubit, nnoise):
        result = 0
        #q = Queue()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        # F_list = Parallel(n_jobs=16, prefer="processes", verbose=5) (delayed(self.noise_assign)(str, idj, nqubit, nnoise) for idj in range(4 ** nnoise))
        # for idj in range(4 ** nnoise):
        #     F_list.append(self.noise_assign(str, idj, nqubit, nnoise))
        procs = []
        F_list = []
        cnt = 0
        max_process_num = 256

        for idj in range(4 ** nnoise):
            proc = Process(target=self.noise_assign, args=(str, idj, nqubit, nnoise, return_dict))
            procs.append(proc)
            proc.start()
            cnt += 1
            if cnt == max_process_num:
                for proc in procs:
                    proc.join()
                cnt = 0
                procs = []
                print("256 subtasks done!")
        for proc in procs:
            proc.join()
        
        for idj in range(4 ** nnoise):
            F_list.append(return_dict[idj])
        # print(len(F_list))

        procs = []
        cnt = 0
        value_dict = manager.dict()

        for ek in range(4 ** nnoise):
            proc = Process(target=self.measure_Ek, args=(ek, nnoise, F_list, value_dict))
            procs.append(proc)
            proc.start()
            cnt += 1
            if cnt == max_process_num:
                for proc in procs:
                    proc.join()
                cnt = 0
                procs = []
                # print("256 subtasks done!")
                result += sum(value_dict.values())
                value_dict = manager.dict()
        for proc in procs:
            proc.join()
        result += sum(value_dict.values())
        return result

    def measure_Ek(self, ek_num, nnoise, F_list, result_dict):
        id = ek_num
        coef = [[[0, 0], [0, 0]] for i in range(nnoise)]
        result = 0
        for j in range(nnoise):
            tmp = id % 4
            if tmp == 0:
                coef[j][0][0] = 1 - self.p
                coef[j][1][1] = 1 - self.p
            elif tmp == 1:
                coef[j][0][1] = self.p / 3
                coef[j][1][0] = self.p / 3
            elif tmp == 2:
                coef[j][0][1] = -1.0j / 3 * self.p
                coef[j][1][0] = 1.0j / 3 * self.p
            elif tmp == 3:
                coef[j][0][0] = 1 / 3 * self.p
                coef[j][1][1] = -1 / 3 * self.p
            id = (id - tmp) / 4

        for i in range(4 ** nnoise):
            num = i
            tmp_coef = 1.0
            for j in range(nnoise):
                tmp = num % 4
                if tmp == 0:
                    tmp_coef *= coef[j][0][0]
                if tmp == 1:
                    tmp_coef *= coef[j][0][1]
                if tmp == 2:
                    tmp_coef *= coef[j][1][0]
                if tmp == 3:
                    tmp_coef *= coef[j][1][1]
                num = int((num - tmp) / 4)

            co = F_list[ek_num]
            result += co * tmp_coef

        result_dict[ek_num] = result
        # return result


def Car(A, B, C):
    return (A & B) | (A | B) & C

# def xo(A, B):
#     return (A & ~B) | (~A & B)

# def Sum(A, B, C):
#     return xo(xo(A,B), C)

def Sum(A, B, C):
    return bdd.apply('xor', bdd.apply('xor', A, B), C)

# def Sum(A, B, C):
#     return (A & B & C) | (~A & ~B & C) | (~A & B & ~C) | (A & ~B & ~C)

# def Sum(A, B, C):
#     return ((A & B | ~A & ~B) & C) | ((~A & B | A & ~B) & ~C)

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

def glb_noise_assign(foo, str, idj, nqubit, nnoise):
    print(psutil.Process().cpu_num())
    dic = {}
    num = idj
    for j in range(nnoise):
        tmp = num % 4
        if tmp == 0:
            dic[r'q{i}'.format(i=foo.nd[j][0])] = False
            dic[r'q{i}'.format(i=foo.nd[j][1])] = False
        if tmp == 1:
            dic[r'q{i}'.format(i=foo.nd[j][0])] = False
            dic[r'q{i}'.format(i=foo.nd[j][1])] = True
        if tmp == 2:
            dic[r'q{i}'.format(i=foo.nd[j][0])] = True
            dic[r'q{i}'.format(i=foo.nd[j][1])] = False
        if tmp == 3:
            dic[r'q{i}'.format(i=foo.nd[j][0])] = True
            dic[r'q{i}'.format(i=foo.nd[j][1])] = True
        num = int((num - tmp) / 4)
    for j in range(nqubit):
        dic[r'q{i}'.format(i=foo.qd[j])] = (str[j] == '1')

    a = to_int_list([bdd.to_expr(bdd.let(dic, foo.Fa[j])) for j in range(foo.r)])
    b = to_int_list([bdd.to_expr(bdd.let(dic, foo.Fb[j])) for j in range(foo.r)])
    c = to_int_list([bdd.to_expr(bdd.let(dic, foo.Fc[j])) for j in range(foo.r)])
    d = to_int_list([bdd.to_expr(bdd.let(dic, foo.Fd[j])) for j in range(foo.r)])

    num_a = list_to_num(a)
    num_b = list_to_num(b)
    num_c = list_to_num(c)
    num_d = list_to_num(d)

    result = ((num_d - num_b) + (num_c - num_a) * 1.0j) / (np.sqrt(2)) ** foo.k
    return result

def cir_test(cir, output_target, input, output):
    dag = circuit_to_dag(cir)
    t_start = time.time()
    nqubits = cir.num_qubits
    gate_num = cir.size()
    depth = cir.depth()
    noise_num = 2
    try:
        with open(output_target, "a") as f:
            f.write(str(nqubits) + "\t" + str(gate_num) + "\t" + str(depth) + "\t" + str(noise_num) + "\t")
        with time_limit(3600):
            test_cir_bdd = cirDD(nqubits, noise_num, input)
            test_cir_bdd.apply_cir(dag, nqubits, noise_num)
            # test_cir_bdd.apply_bv_cir(dag, nqubits, noise_num)
            build_time = time.time() - t_start
            result = test_cir_bdd.noise_measure(output, nqubits, noise_num)
            run_time = time.time() - t_start
            bdd_num = len(test_cir_bdd.Fa)
            with open(output_target, "a") as f:
                f.write(str(bdd_num) + "\t" + str(build_time) + "\t" + str(run_time) + "\t" + str(result) + "\n")
    except TimeoutException as e:
        with open(output_target, 'a') as f:
            f.write(str(e)+"\n")
    except Exception as e:
        with open(output_target, 'a') as f:
            f.write(str(e)+"\n")

def file_test(path, file_name, output_target):
    print(file_name)
    with open(output_target, "a") as f:
        f.write(file_name + "\t")
    try:
        cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
        nqubits = cir.num_qubits
        input = '0'*nqubits
        output = '1'*nqubits
        cir_test(cir, output_target, input, output)
    except Exception as e:
        with open(output_target, 'a') as f:
            f.write(str(e) + "\n")

def folder_test(folder_path, output_target):
    folder = os.listdir(folder_path)
    for f in folder:
        file_test(folder_path, f, output_target)



def server_test(server_num):
    path = 'tmp_test_panda%d/'%server_num
    folder_test(path, "BDD_result_panda%d.txt"%server_num)

if __name__ == "__main__":
    bdd = _bdd.BDD()
    server_test(22)



