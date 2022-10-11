from my import *

def run1(cir, error_pos, test_num = 1):
    run_time = 0
    result = 0
    for loop in range(test_num):
        t_start = time.time()
        result += alg1(cir, error_pos)
        run_time = run_time + time.time() - t_start
    run_time = run_time / test_num
    print('run_time1:', run_time)
    print('result1:', result)

def run2(cir, error_pos, test_num = 1):
    run_time = 0
    result = 0
    for loop in range(test_num):
        t_start = time.time()
        result += alg2(cir, error_pos)
        run_time = run_time + time.time() - t_start
    run_time = run_time / test_num
    print('run_time2:', run_time)
    print('result2:', result)

if __name__ == '__main__':
    path = 'Benchmarks/'
    file_name = 'quantum_volume_n10_d10_i0.qasm'
    cir = CreateCircuitFromQASM(file_name, path)
    print('Qubits: ', cir.num_qubits)
    print('Gates: ', cir.size())
    test_num = 1
    # error_num = 0 will randomly generate error_num
    error_num = 3
    error_pos = generate_error(cir.size(), error_num)
    print(file_name)
    #run1(cir, error_pos, test_num)
    run2(cir, error_pos, test_num)

