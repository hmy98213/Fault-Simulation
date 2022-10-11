import random


def header(nq, nc):
    str = ""
    str += "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
    str += "qreg qr[%d];\n"%(nq)
    str += "creg cr[%d];\n"%(nc)
    return str

def cnot(n, m, k):
    str = header(n, n)
    cnt = 0

    for i in range(n):
        str += "h qr[%d];\n"%i

    while cnt < m:
        flag = random.randint(0, 1)
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x==y:
            continue
        cnt += 1
        if flag == 1:
            str += "cx qr[%d], qr[%d];\n"%(x, y)
        else:
            str += "h qr[%d];\n"%x
    filename = "./approximate_test/cx_h%d_%d_%d.qasm"%(n, m, k)
    file = open(filename, 'w')
    file.write(str)
    file.close()

if __name__ == '__main__':
    # for i in range(10,50, 5):
    #     cnot(i)\
    #cnot(40)
    for i in range(20, 85, 5):
    #    for k in range(1, 11, 1):
        cnot(i, i*8, 0)