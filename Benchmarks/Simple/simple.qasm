OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry(0.8) q[0];
rx(0.5) q[0];
cx q[0],q[1];
ry(0.9) q[0];
rx(1.2) q[0];