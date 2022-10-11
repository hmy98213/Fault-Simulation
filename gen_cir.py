

def gen_bv(hstr):
    nqubit = len(hstr)
    qasm_txt=''
    qasm_txt+="OPENQASM 3\n"
    qasm_txt+="qubit qr["+str(nqubit+1)+"]\nbit cr["+str(nqubit)+"]\n"
    for i in range(nqubit):
        qasm_txt+="h qr["+str(i)+"]\n"
    qasm_txt+="x qr["+str(nqubit)+"]\nh qr["+str(nqubit)+"]\n"
    for i in range(nqubit):
        qasm_txt += "x qr[" + str(nqubit) + "]\nh qr[" + str(nqubit) + "]\n"
