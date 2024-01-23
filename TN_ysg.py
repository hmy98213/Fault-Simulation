from tn_construction import *

if __name__ =='__main__':
    f = 'ysg10.qasm'
    cir_tn = QCTN(f, '')
    output_file = 'ysg10.tn'
    cir_tn.io_test(output_file, 0, random_pos=True)
    print('This program is being run by itself')