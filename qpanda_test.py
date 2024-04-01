from numpy import pi
from pyqpanda import *

# 构建量子虚拟机
qvm = MPSQVM()

# 初始化操作
qvm.set_configure(101, 101)
qvm.init_qvm()



nq = 50
l = 5


q = qvm.qAlloc_many(nq)
c = qvm.cAlloc_many(nq)

# 构建量子程序
prog = QProg()
prog << hadamard_circuit(q)

for _ in range(l):

    for i in range(nq):
        prog << T(q[i])
    for j in range(nq-1):
        prog << (CNOT(q[j], q[j+1]))

prog.insert(measure_all(q, c))
    # prog << T(q[2], q[4])
    # prog << CZ(q[3], q[7])
    # prog << CNOT(q[0], q[1])
    
    # << Measure(q[0], c[0])\
    # << Measure(q[1], c[1])\
    # << Measure(q[2], c[2])\
    # << Measure(q[3], c[3])

t_start = time.time()
result = qvm.pmeasure_bin_index(prog,'0'*nq)
run_time = time.time() - t_start
# 打印量子态在量子程序多次运行结果中出现的次数
print(result)
print(run_time)
qvm.finalize()