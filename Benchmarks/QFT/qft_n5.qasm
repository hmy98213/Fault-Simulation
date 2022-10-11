OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
h q[0];
u1(0.785398163397448) q[1];
cx q[1],q[0];
u1(-0.785398163397448) q[0];
cx q[1],q[0];
u1(0.785398163397448) q[0];
h q[1];
u1(0.392699081698724) q[2];
cx q[2],q[0];
u1(-0.392699081698724) q[0];
cx q[2],q[0];
u1(0.392699081698724) q[0];
u1(0.785398163397448) q[2];
cx q[2],q[1];
u1(-0.785398163397448) q[1];
cx q[2],q[1];
u1(0.785398163397448) q[1];
h q[2];
u1(0.196349540849362) q[3];
cx q[3],q[0];
u1(-0.196349540849362) q[0];
cx q[3],q[0];
u1(0.196349540849362) q[0];
u1(0.392699081698724) q[3];
cx q[3],q[1];
u1(-0.392699081698724) q[1];
cx q[3],q[1];
u1(0.392699081698724) q[1];
u1(0.785398163397448) q[3];
cx q[3],q[2];
u1(-0.785398163397448) q[2];
cx q[3],q[2];
u1(0.785398163397448) q[2];
h q[3];
u1(0.0981747704246810) q[4];
cx q[4],q[0];
u1(-0.0981747704246810) q[0];
cx q[4],q[0];
u1(0.0981747704246810) q[0];
u1(0.196349540849362) q[4];
cx q[4],q[1];
u1(-0.196349540849362) q[1];
cx q[4],q[1];
u1(0.196349540849362) q[1];
u1(0.392699081698724) q[4];
cx q[4],q[2];
u1(-0.392699081698724) q[2];
cx q[4],q[2];
u1(0.392699081698724) q[2];
u1(0.785398163397448) q[4];
cx q[4],q[3];
u1(-0.785398163397448) q[3];
cx q[4],q[3];
u1(0.785398163397448) q[3];
h q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
