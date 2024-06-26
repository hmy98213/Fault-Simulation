// Generated from Cirq v1.0.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0, 5), q(0, 6), q(1, 4), q(1, 5), q(1, 6), q(1, 7), q(2, 3), q(2, 4), q(2, 5), q(2, 6)]
qreg q[10];
creg m0[1];  // Measurement: q(0, 5)
creg m1[1];  // Measurement: q(2, 4)
creg m2[1];  // Measurement: q(2, 5)
creg m3[1];  // Measurement: q(1, 5)
creg m4[1];  // Measurement: q(1, 7)
creg m5[1];  // Measurement: q(2, 6)
creg m6[1];  // Measurement: q(1, 4)
creg m7[1];  // Measurement: q(2, 3)
creg m8[1];  // Measurement: q(1, 6)
creg m9[1];  // Measurement: q(0, 6)


h q[0];
h q[7];
h q[8];
h q[3];
h q[5];
h q[9];
h q[2];
h q[6];
h q[4];
h q[1];

// Gate: ZZ**3.5335063371251203
rz(pi*3.5335063371) q[0];
rz(pi*3.5335063371) q[1];
u3(pi*0.5,0,0) q[0];
u3(pi*0.5,0,pi*1.5) q[1];
sx q[0];
cx q[0],q[1];
rx(pi*0.0335063371) q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
sxdg q[1];
s q[1];
cx q[0],q[1];
u3(pi*0.5,pi*1.4664936629,pi*1.0) q[0];
u3(pi*0.5,pi*1.9664936629,pi*1.0) q[1];

// Gate: ZZ**0.6911503837897546
rz(pi*0.6911503838) q[7];
rz(pi*0.6911503838) q[2];
u3(pi*0.5,0,pi*0.25) q[7];
u3(pi*0.5,0,pi*1.75) q[2];
sx q[7];
cx q[7],q[2];
rx(pi*0.1911503838) q[7];
ry(pi*0.5) q[2];
cx q[2],q[7];
sxdg q[2];
s q[2];
cx q[7],q[2];
u3(pi*0.5,pi*1.0588496162,pi*1.0) q[7];
u3(pi*0.5,pi*1.5588496162,pi*1.0) q[2];

// Gate: ZZ**1.650121541298039
rz(pi*1.6501215413) q[0];
rz(pi*1.6501215413) q[3];
u3(pi*0.5,pi*1.0,pi*1.25) q[0];
u3(pi*0.5,pi*1.0,pi*0.75) q[3];
sx q[0];
cx q[0],q[3];
rx(pi*0.1501215413) q[0];
ry(pi*0.5) q[3];
cx q[3],q[0];
sxdg q[3];
s q[3];
cx q[0],q[3];
u3(pi*0.5,pi*0.0998784587,0) q[0];
u3(pi*0.5,pi*0.5998784587,0) q[3];

// Gate: ZZ**0.7861835615608459
rz(pi*0.7861835616) q[7];
rz(pi*0.7861835616) q[6];
u3(pi*0.5,pi*1.0,pi*0.6033374354) q[7];
u3(pi*0.5,pi*1.0,pi*1.1033374354) q[6];
sx q[7];
cx q[7],q[6];
rx(pi*0.2861835616) q[7];
ry(pi*0.5) q[6];
cx q[6],q[7];
sxdg q[6];
s q[6];
cx q[7],q[6];
u3(pi*0.5,pi*0.610479003,0) q[7];
u3(pi*0.5,pi*0.110479003,0) q[6];

// Gate: ZZ**2.9114709917143413
rz(pi*2.9114709917) q[7];
rz(pi*2.9114709917) q[8];
u3(pi*0.5,pi*1.0,pi*1.4550477507) q[7];
u3(pi*0.5,pi*1.0,pi*1.9550477507) q[8];
sx q[7];
cx q[7],q[8];
rx(pi*0.4114709917) q[7];
ry(pi*0.5) q[8];
cx q[8],q[7];
sxdg q[8];
s q[8];
cx q[7],q[8];
u3(pi*0.5,pi*1.6334812576,0) q[7];
u3(pi*0.5,pi*1.1334812576,0) q[8];

// Gate: ZZ**3.792687731046278
rz(pi*3.792687731) q[8];
rz(pi*3.792687731) q[3];
u3(pi*0.5,0,pi*1.8664773743) q[8];
u3(pi*0.5,0,pi*1.3664773743) q[3];
sx q[8];
cx q[8],q[3];
rx(pi*0.292687731) q[8];
ry(pi*0.5) q[3];
cx q[3],q[8];
sxdg q[3];
s q[3];
cx q[8],q[3];
u3(pi*0.5,pi*1.3408348947,pi*1.0) q[8];
u3(pi*0.5,pi*1.8408348947,pi*1.0) q[3];

// Gate: ZZ**2.3239931654930497
rz(pi*2.3239931655) q[8];
rz(pi*2.3239931655) q[9];
u3(pi*0.5,0,pi*0.25) q[8];
u3(pi*0.5,pi*1.0,pi*0.75) q[9];
sx q[8];
cx q[8],q[9];
rx(pi*0.1760068345) q[8];
ry(pi*0.5) q[9];
cx q[9],q[8];
sxdg q[9];
s q[9];
cx q[8],q[9];
u3(pi*0.5,pi*0.4260068345,pi*1.0) q[8];
u3(pi*0.5,pi*1.9260068345,0) q[9];

// Gate: ZZ**2.8682740927274812
rz(pi*2.8682740927) q[3];
rz(pi*2.8682740927) q[2];
u3(pi*0.5,pi*1.0,pi*1.5) q[3];
u3(pi*0.5,pi*1.0,pi*1.0) q[2];
sx q[3];
cx q[3],q[2];
rx(pi*0.3682740927) q[3];
ry(pi*0.5) q[2];
cx q[2],q[3];
sxdg q[2];
s q[2];
cx q[3],q[2];
u3(pi*0.5,pi*1.6317259073,0) q[3];
u3(pi*0.5,pi*0.1317259073,0) q[2];

// Gate: ZZ**1.3909401473768812
rz(pi*1.3909401474) q[3];
rz(pi*1.3909401474) q[4];
u3(pi*0.5,0,0) q[3];
u3(pi*0.5,pi*1.0,pi*1.5) q[4];
sx q[3];
cx q[3],q[4];
rx(pi*0.1090598526) q[3];
ry(pi*0.5) q[4];
cx q[4],q[3];
sxdg q[4];
rz(pi*0.5) q[4];
cx q[3],q[4];
u3(pi*0.5,pi*0.6090598526,pi*1.0) q[3];
u3(pi*0.5,pi*1.1090598526,0) q[4];

// Gate: ZZ**0.6133959656134071
rz(pi*0.6133959656) q[5];
rz(pi*0.6133959656) q[4];
u3(pi*0.5,pi*1.0,pi*1.25) q[5];
u3(pi*0.5,pi*1.0,pi*0.75) q[4];
sx q[5];
cx q[5],q[4];
rx(pi*0.1133959656) q[5];
ry(pi*0.5) q[4];
cx q[4],q[5];
sxdg q[4];
s q[4];
cx q[5],q[4];
u3(pi*0.5,pi*0.1366040344,0) q[5];
u3(pi*0.5,pi*0.6366040344,0) q[4];

// Gate: ZZ**2.920110371511713
rz(pi*2.9201103715) q[9];
rz(pi*2.9201103715) q[4];
u3(pi*0.5,pi*1.0,pi*0.25) q[9];
u3(pi*0.5,pi*1.0,pi*0.75) q[4];
sx q[9];
cx q[9],q[4];
rx(pi*0.4201103715) q[9];
ry(pi*0.5) q[4];
cx q[4],q[9];
sxdg q[4];
s q[4];
cx q[9],q[4];
u3(pi*0.5,pi*0.8298896285,0) q[9];
u3(pi*0.5,pi*0.3298896285,0) q[4];

// Gate: ZZ**2.419026343264141
rz(pi*2.4190263433) q[4];
rz(pi*2.4190263433) q[1];
u3(pi*0.5,0,pi*0.25) q[4];
u3(pi*0.5,pi*1.0,pi*0.75) q[1];
sx q[4];
cx q[4],q[1];
rx(pi*0.0809736567) q[4];
ry(pi*0.5) q[1];
cx q[1],q[4];
sxdg q[1];
s q[1];
cx q[4],q[1];
u3(pi*0.5,pi*0.3309736567,pi*1.0) q[4];
u3(pi*0.5,pi*1.8309736567,0) q[1];

rx(pi*1.2566370614) q[0];
rx(pi*1.2566370614) q[7];
rx(pi*1.2566370614) q[8];
rx(pi*1.2566370614) q[3];
rx(pi*1.2566370614) q[5];
rx(pi*1.2566370614) q[9];
rx(pi*1.2566370614) q[2];
rx(pi*1.2566370614) q[6];
rx(pi*1.2566370614) q[4];
rx(pi*1.2566370614) q[1];
measure q[0] -> m0[0];
measure q[7] -> m1[0];
measure q[8] -> m2[0];
measure q[3] -> m3[0];
measure q[5] -> m4[0];
measure q[9] -> m5[0];
measure q[2] -> m6[0];
measure q[6] -> m7[0];
measure q[4] -> m8[0];
measure q[1] -> m9[0];
