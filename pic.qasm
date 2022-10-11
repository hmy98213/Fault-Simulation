// Generated from Cirq v1.0.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0, 0), q(0, 1)]
qreg q[2];


h q[0];
h q[1];

// Gate: ZZ**-1.0
rz(pi*-1.0) q[0];
rz(pi*-1.0) q[1];
u3(0,0,0) q[0];
u3(0,0,0) q[1];
sx q[0];
cx q[0],q[1];
sx q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
sxdg q[1];
s q[1];
cx q[0],q[1];
u3(0,0,0) q[0];
u3(0,0,0) q[1];

rx(pi*1.0) q[0];
rx(pi*1.0) q[1];