OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
s q[0];
s q[1];
s q[4];
h q[4];
s q[7];
h q[7];
s q[7];
s q[8];
h q[11];
cx q[0],q[1];
cx q[0],q[8];
cx q[0],q[11];
cx q[2],q[0];
cx q[6],q[0];
cx q[9],q[10];
cx q[0],q[9];
h q[9];
cx q[9],q[0];
cx q[7],q[4];
cx q[4],q[0];
cx q[0],q[7];
h q[1];
s q[1];
h q[3];
s q[4];
h q[6];
s q[9];
swap q[7],q[1];
cx q[7],q[5];
cx q[6],q[7];
cx q[11],q[7];
cx q[1],q[7];
cx q[2],q[9];
cx q[2],q[10];
cx q[7],q[2];
h q[2];
cx q[2],q[7];
cx q[4],q[3];
cx q[3],q[7];
cx q[7],q[4];
h q[2];
h q[3];
s q[3];
h q[4];
s q[5];
s q[6];
h q[6];
s q[9];
h q[9];
h q[10];
s q[10];
h q[11];
swap q[2],q[1];
cx q[2],q[11];
cx q[2],q[1];
cx q[9],q[2];
cx q[2],q[4];
h q[4];
cx q[4],q[2];
cx q[5],q[3];
cx q[3],q[2];
cx q[2],q[5];
cx q[10],q[6];
cx q[6],q[2];
cx q[2],q[10];
s q[1];
h q[1];
s q[4];
h q[4];
s q[5];
h q[5];
h q[9];
h q[11];
swap q[6],q[1];
cx q[6],q[3];
cx q[6],q[11];
cx q[9],q[6];
cx q[10],q[6];
cx q[6],q[8];
h q[8];
cx q[8],q[6];
cx q[5],q[4];
cx q[4],q[6];
cx q[6],q[5];
h q[1];
h q[3];
s q[3];
s q[4];
s q[8];
h q[8];
s q[9];
h q[9];
s q[11];
h q[11];
s q[11];
cx q[8],q[1];
cx q[10],q[1];
cx q[4],q[3];
cx q[3],q[1];
cx q[1],q[4];
cx q[11],q[9];
cx q[9],q[1];
cx q[1],q[11];
s q[3];
h q[3];
s q[3];
s q[4];
h q[4];
h q[5];
s q[8];
h q[8];
s q[8];
h q[10];
s q[11];
cx q[3],q[5];
cx q[3],q[10];
cx q[4],q[3];
cx q[3],q[9];
h q[9];
cx q[9],q[3];
cx q[11],q[8];
cx q[8],q[3];
cx q[3],q[11];
s q[9];
h q[9];
s q[10];
s q[11];
h q[11];
swap q[9],q[11];
cx q[9],q[10];
cx q[11],q[9];
h q[4];
s q[4];
s q[5];
h q[5];
s q[5];
s q[8];
h q[8];
h q[10];
s q[11];
h q[11];
swap q[10],q[4];
cx q[11],q[10];
cx q[10],q[4];
h q[4];
cx q[4],q[10];
cx q[8],q[5];
cx q[5],q[10];
cx q[10],q[8];
s q[4];
h q[4];
s q[5];
h q[5];
s q[5];
cx q[11],q[8];
cx q[5],q[4];
cx q[4],q[8];
cx q[8],q[5];
s q[4];
h q[5];
h q[11];
cx q[5],q[11];
cx q[4],q[5];
h q[5];
cx q[5],q[4];
swap q[11],q[5];
cx q[11],q[5];
h q[5];
s q[5];
x q[0];
y q[1];
z q[2];
y q[3];
z q[5];
z q[6];
x q[8];
z q[9];
x q[10];
x q[11];
