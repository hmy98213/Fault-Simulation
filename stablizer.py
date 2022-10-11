import numpy as np

from my_cpu import *

dic_pauli={'I':[[1, 0], [0, 1]], 'X':[[0, 1], [1, 0]], 'Y':[[0, -1.0j], [1.0j, 0]], 'Z':[[1, 0], [0, -1]]}

class pauli:
    def __init__(self, op, coef):
        self.op = op
        self.coef = coef

class linear_pauli:
    def __init__(self, pauli_elements):
        self.pauli_elements = pauli_elements

class stab_proj:
    def __init__(self, stablizers, coef):
        self.stablizers = stablizers
        self.coef = coef
    def check_matrix(self):
        l = len(self.stablizers[0].op)
        mat = [[0]*2*l]*len(self.stablizers)
        cnt = 0
        for s in self.stablizers:
            for i in range(l):
                if s.op[i] == 'I':
                    mat[cnt][i] = 0
                    mat[cnt][i+l] = 0
                if s.op[i] == 'x':
                    mat[cnt][i] = 1
                    mat[cnt][i+l] = 0
                if s.op[i] == 'y':
                    mat[cnt][i] = 1
                    mat[cnt][i+l] = 1
                if s.op[i] == 'z':
                    mat[cnt][i] = 0
                    mat[cnt][i+l] = 1
            cnt += 1
        return mat


class pauli_operation:
    # trace(g*h), where g and h are pauli matrix
    def mul(g, h):
        l = len(g.op)
        result_op = ''
        sign = 1.0
        for i in range(l):
            if g.op[i] == "I":
                result_op+=h.op[i]
            elif h.op[i] == 'I':
                result_op+=g.op[i]
            elif g.op[i] ==h.op[i]:
                result_op+='I'
            elif g.op[i]=='x' and h.op[i]=='y':
                sign*=1.0j
                result_op+='z'
            elif g.op[i]=='y' and h.op[i]=='z':
                sign*=1.0j
                result_op+='x'
            elif g.op[i]=='z' and h.op[i]=='x':
                sign*=1.0j
                result_op+='y'
            elif g.op[i]=='y' and h.op[i]=='x':
                sign*=-1.0j
                result_op+='z'
            elif g.op[i]=='z' and h.op[i]=='y':
                sign*=-1.0j
                result_op+='z'
            elif g.op[i]=='x' and h.op[i]=='z':
                sign*=-1.0j
                result_op+='z'
        return pauli(result_op, sign*g.coef*h.coef)

    def trace(g, h):
        l = len(g.op)
        for i in range(l):
            if g.op[i] != h.op[i]:
                return 0
        return g.coef * h.coef * 2**l

    def apply_clifford(unitary, g, qubit):
        if g.op[qubit] == 'I':
            return pauli(g.op, g.coef)
        if unitary == 'x':
            if g.op[qubit] == 'x':
                return pauli(g.op, g.coef)
            if g.op[qubit] == 'y':
                return pauli(g.op, -g.coef)
            if g.op[qubit] == 'z':
                return pauli(g.op, -g.coef)
        if unitary == 'y':
            if g.op[qubit] == 'x':
                return pauli(g.op, -g.coef)
            if g.op[qubit] == 'y':
                return pauli(g.op, g.coef)
            if g.op[qubit] == 'z':
                return pauli(g.op, -g.coef)
        if unitary == 'z':
            if g.op[qubit] == 'x':
                return pauli(g.op, -g.coef)
            if g.op[qubit] == 'y':
                return pauli(g.op, -g.coef)
            if g.op[qubit] == 'z':
                return pauli(g.op, g.coef)
        if unitary == 'h':
            if g.op[qubit] == 'x':
                op = g.op[:qubit]+'z'+g.op[qubit+1:]
                return pauli(''.join(op), g.coef)
            if g.op[qubit] == 'y':
                return pauli(g.op, -g.coef)
            if g.op[qubit] == 'z':
                op = g.op[:qubit]+'x'+g.op[qubit+1:]
                return pauli(''.join(op), g.coef)
        if unitary == 's':
            if g.op[qubit] == 'x':
                op = g.op[:qubit]+'y'+g.op[qubit+1:]
                return pauli(''.join(op), g.coef)
            if g.op[qubit] == 'y':
                op = g.op[:qubit]+'x'+g.op[qubit+1:]
                return pauli(''.join(op), -g.coef)
            if g.op[qubit] == 'z':
                return pauli(g.op, g.coef)

    def apply_CNOT(g, i, j):
        if g.op[i] == 'I' and g.op[j] == 'I':
            return pauli(g.op, g.coef)
        if g.op[i] == 'I' and g.op[j] == 'x':
            return pauli(g.op, g.coef)
        if g.op[i] == 'I' and g.op[j] == 'y':
            op = list(g.op)
            op[i] = 'z'
            op[j] = 'y'
            return pauli(''.join(op), g.coef)
        if g.op[i] == 'I' and g.op[j] == 'z':
            op = list(g.op)
            op[i] = 'z'
            op[j] = 'z'
            return pauli(''.join(op), g.coef)
        if g.op[i] == 'x' and g.op[j] == 'I':
            op = list(g.op)
            op[i] = 'x'
            op[j] = 'x'
            return pauli(''.join(op), g.coef)
        if g.op[i] == 'x' and g.op[j] == 'x':
            op = list(g.op)
            op[i] = 'x'
            op[j] = 'I'
            return pauli(''.join(op), g.coef)
        if g.op[i] == 'x' and g.op[j] == 'y':
            op = list(g.op)
            op[i] = 'y'
            op[j] = 'z'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'x' and g.op[j] == 'z':
            op = list(g.op)
            op[i] = 'y'
            op[j] = 'y'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'y' and g.op[j] == 'I':
            op = list(g.op)
            op[i] = 'y'
            op[j] = 'x'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'y' and g.op[j] == 'x':
            op = list(g.op)
            op[i] = 'y'
            op[j] = 'I'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'y' and g.op[j] == 'y':
            op = list(g.op)
            op[i] = 'x'
            op[j] = 'z'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'y' and g.op[j] == 'z':
            op = list(g.op)
            op[i] = 'x'
            op[j] = 'y'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'z' and g.op[j] == 'I':
            return pauli(g.op, g.coef)
        if g.op[i] == 'z' and g.op[j] == 'x':
            return pauli(g.op, g.coef)
        if g.op[i] == 'z' and g.op[j] == 'y':
            op = list(g.op)
            op[i] = 'I'
            op[j] = 'y'
            return pauli(''.join(op), -g.coef)
        if g.op[i] == 'z' and g.op[j] == 'z':
            op = list(g.op)
            op[i] = 'I'
            op[j] = 'z'
            return pauli(''.join(op), g.coef)

    def apply_T(g, qubit):
        if g.op[qubit] == 'I':
            return linear_pauli(pauli(g.op, g.coef))
        if g.op[qubit] == 'x':
            op = list(g.op)
            op[qubit] = 'y'
            return linear_pauli([pauli(g.op, g.coef / np.sqrt(2)), pauli(''.join(op), g.coef / np.sqrt(2))])
        if g.op[qubit] == 'y':
            op = list(g.op)
            op[qubit] = 'x'
            return linear_pauli(pauli(g.op, -g.coef / np.sqrt(2)), pauli(''.join(op), g.coef / np.sqrt(2)))
        if g.op[qubit] == 'z':
            return linear_pauli([pauli(g.op, g.coef / np.sqrt(2))])

class linear_pauli_operation:
    def add(s1, s2):
        result = s1.pauli_elements[:]
        for e2 in s2.pauli_elements:
            flag = 0
            for e in result:
                if e2.op == e.op:
                    sum = e2.coef+e.coef
                    result.remove(e)
                    result.append(pauli(e.op, sum))
                    flag = 1
                    break
            if flag ==0:
                result.append(e2)
        return linear_pauli(result)

    def mul(s1, s2):
        result = linear_pauli([])
        for i in s1.pauli_elements:
            for j in s2.pauli_elements:
                if i is None or j is None:
                    continue
                result = linear_pauli_operation.add(result, linear_pauli([pauli_operation.mul(i, j)]))
        return result

    # sum of trace in s1, s2, with c1, c2 as coefficient
    def trace(s1, s2):
        result = 0
        for i in s1.pauli_elements:
            for j in s2.pauli_elements:
                if i is None or j is None:
                    continue
                result += pauli_operation.trace(i, j)
        return result

    def apply_clifford(unitary, s, qubit):
        result = []
        for e in s.pauli_elements:
            result.append(pauli_operation.apply_clifford(unitary, e, qubit))
        return linear_pauli(result)
    def apply_CNOT(s, i, j):
        result = []
        for e in s.pauli_elements:
            result.append(pauli_operation.apply_CNOT(e, i, j))
        return linear_pauli(result)
    def apply_T(s, qubit):
        result = []
        for e in s.pauli_elements:
            result.append(pauli_operation.apply_T(e, qubit).pauli_elements)
        return linear_pauli(result)
    def apply_circuit(s, cir_dag):
        result = s
        for node in cir_dag.topological_op_nodes():
            q = [x.index for x in node.qargs]
            if node.name == 'cx':
                result = linear_pauli_operation.apply_CNOT(result, q[0], q[1])
            elif node.name =='t':
                result = linear_pauli_operation.apply_T(result, q[0])
            else:
                result = linear_pauli_operation.apply_clifford(node.name, result, q[0])
        return result


class stab_proj_operations:
    def apply_clifford(unitary, sp, qubit):
        tmp = linear_pauli(sp.stablizers)
        result = linear_pauli_operation.apply_clifford(unitary, tmp, qubit)
        return stab_proj(result.pauli_elements, sp.coef)
    def apply_CNOT(sp, i, j):
        tmp = linear_pauli(sp.stablizers)
        result = linear_pauli_operation.apply_CNOT(tmp, i, j)
        return stab_proj(result.pauli_elements, sp.coef)
    def apply_T(sp, qubit):
        pass
    def apply_circuit(sp, cir_dag):
        result = sp
        for node in cir_dag.topological_op_nodes():
            q = [x.index for x in node.qargs]
            if node.name == 'cx':
                result = stab_proj_operations.apply_CNOT(result, q[0], q[1])
            else:
                result = stab_proj_operations.apply_clifford(node.name, result, q[0])
        return result
    def check_comu(sp, sq):
        l = len(sp.stablizers[0].op)
        A = np.array(sp.check_matrix(), dtype=int)
        B = np.array(sq.check_matrix(), dtype=int)
        I = np.identity(l, dtype=int)
        Z = np.zeros((l, l), dtype=int)
        M = np.block([[Z, I], [I, Z]])
        foo = np.matmul(np.matmul(A, M), B.T)
        result = foo % 2
        return np.array_equal(result, np.zeros((l, l)))
    def trace(sp, sq):
        l = len(sp.stablizers[0].op)
        if not stab_proj_operations.check_comu(sp, sq):
            return 0
        A = np.array(sp.check_matrix(), dtype=int)
        B = np.array(sq.check_matrix(), dtype=int)
        mat = np.block([[A], [B]])
        result = np.linalg.matrix_rank(mat)
        return 2.0**(result-l)


def stab_to_proj(s):
    vec = []
    for w in s:
        vec.append(tn.Node(np.array(dic_pauli[w], dtype=complex)))

#todo
def stabset_to_lp(s):
    l = len(s[0])
    if s[0].startswith('-'):
        l = l-1
    result = linear_pauli([pauli('I'*l, 1.0)])
    for str in s:
        flag = 1.0
        if str.startswith('-'):
            flag = -1.0
            str = str[1:]
        tmp = linear_pauli_operation.add(linear_pauli([pauli('I'*l, 0.5)]), linear_pauli([pauli(str, 0.5*flag)]))
        result = linear_pauli_operation.mul(result, tmp)
    return result

def stabset_to_sp(s):
    result = []
    l = len(s[0])
    for str in s:
        if str.startswith('-'):
            result.append(pauli(str[1:], -1.0))
        else:
            result.append(pauli(str, 1.0))
    return stab_proj(result, 1.0)


def trace_contract(cir, proj1, proj2):
    all_nodes = []
    with tn.NodeCollection(all_nodes):

        qubits = [node[1] for node in proj2]
        start_wires = [node[0] for node in proj2]
        dag = circuit_to_dag(cir)
        inv_cir = cir.inverse()
        inv_dag = circuit_to_dag(inv_cir)

        dag_to_error_unitary(dag, qubits, 0, 0)
        for i in range(cir.num_qubits):
            tn.connect(qubits[i], proj1[i][0])
        qubits = [node[1] for node in proj1]
        dag_to_error_unitary(inv_dag, qubits, 0, 0)

        for i in range(cir.num_qubits):
            tn.connect(start_wires[i], qubits[i])
    return tn.contractors.auto(all_nodes + proj1 + proj2).tensor

def bv_test(numq):
    sp = []
    sq = []
    for i in range(numq):
        sp.append('I'*i + 'z' + 'I'*(numq-i-1))
        sq.append('-' + 'I'*i + 'z' + 'I'*(numq-i-1))
    return sp, sq

# def bv_stb_test(numq):
#     s, t = bv_test(numq)
#     l1 = []
#     l2 = []
#     for i in range(numq):
#         l1.append(pauli(s[i]))

def my_test():
    path = 'Benchmarks/'
    files = os.listdir(path)
    file_name = 'qmy.qasm'
    cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
    cir_dag = circuit_to_dag(cir)
    P = stabset_to_lp(['-Iz', '-zI'])
    Q = stabset_to_lp(['-Iz', 'zI'])
    tmp = linear_pauli_operation.apply_circuit(P, cir_dag)
    print(linear_pauli_operation.trace(Q, tmp))

if __name__ == '__main__':
    # #stab_to_proj('IXX')
    path = 'Benchmarks/'
    files = os.listdir(path)
    file_name = 'bv_n7.qasm'
    cir, res = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=False)
    cir_dag = circuit_to_dag(cir)
    # g = pauli('Ix', 0.5)
    # h = pauli('Iy', 0.5)
    # lp = linear_pauli([g, h])
    #
    # t = linear_pauli_operation.mul(lp, lp)
    #
    # result = linear_pauli_operation.apply_circuit(lp, cir_dag)
    # print(linear_pauli_operation.apply_circuit(lp, cir_dag))

    # test the result of apply circuit

    my_test()
    sp, sq = bv_test(7)
    P = stabset_to_lp(sp)
    Q = stabset_to_lp(sq)
    tmp = linear_pauli_operation.apply_circuit(P, cir_dag)

    print(linear_pauli_operation.trace(Q, linear_pauli_operation.apply_circuit(P, cir_dag)))

    # nqubits = cir.num_qubits
    #
    # proj1 = [tn.Node(np.array([[1, 0], [0, 0]], dtype = complex)) for i in range(nqubits)]
    # proj2 = [tn.Node(np.array([[0, 0], [0, 1]], dtype = complex)) for i in range(nqubits)]
    #
    # print(trace_contract(cir, proj1, proj2))
    #
    # g = pauli('xy', 0.2)
    # h = pauli('xy', 0.3)
    # ls = linear_pauli_operation.trace(linear_pauli([g, h]), linear_pauli([g, h]))
    # print(ls)

    p = stabset_to_sp(sp)
    q = stabset_to_sp(sq)
    #print(p.check_matrix())
    #print(q.check_matrix())
    tmp = stab_proj_operations.apply_circuit(p, cir_dag)
    print(stab_proj_operations.check_comu(tmp, q))
    print(stab_proj_operations.trace(tmp, q))

