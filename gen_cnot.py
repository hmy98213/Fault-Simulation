import random


def header(nq, nc):
    str = ""
    str += "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
    str += "qreg qr[%d];\n"%(nq)
    str += "creg cr[%d];\n"%(nc)
    return str

def cnot(n):
    str = header(n, n)
    for i in range(n):
        str += "h qr[%d];\n"%i
    # for i in range(n):
    #     flag = random.randint(0, 1)
    #     x = random.randint(0, n-1)
    #     y = random.randint(0, n-1)
    #     if x==y:
    #         continue
    #     if flag == 1:
    #         str += "cx qr[%d], qr[%d];\n"%(x, y)
    #     else:
    #         str += "h qr[%d];\n"%x
    str += "cx qr[%d], qr[%d];\n"%(1, 2)

    filename = "./test/h_30.qasm"
    file = open(filename, 'w')
    file.write(str)
    file.close()

if __name__ == '__main__':
    cnot(30)