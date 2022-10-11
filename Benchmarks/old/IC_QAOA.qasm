OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[12];
h q[4];
h q[2];
h q[3];
h q[14];
h q[11];
h q[7];
h q[9];
h q[16];
h q[19];
h q[10];
h q[5];
h q[17];
cx q[4],q[3];
u3(0,0,-0.98) q[3];
cx q[4],q[3];
cx q[11],q[2];
u3(0,0,-0.98) q[2];
cx q[11],q[2];
cx q[14],q[10];
u3(0,0,-0.98) q[10];
cx q[14],q[10];
cx q[9],q[16];
u3(0,0,-0.98) q[16];
cx q[9],q[16];
cx q[5],q[17];
u3(0,0,-0.98) q[17];
cx q[5],q[17];
cx q[4],q[19];
cx q[19],q[4];
cx q[4],q[19];
cx q[7],q[4];
u3(0,0,-0.98) q[4];
cx q[7],q[4];
cx q[3],q[4];
u3(0,0,-0.98) q[4];
cx q[3],q[4];
cx q[9],q[5];
u3(0,0,-0.98) q[5];
cx q[9],q[5];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[2];
u3(0,0,-0.98) q[2];
cx q[9],q[2];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[6],q[14];
cx q[14],q[6];
cx q[6],q[14];
cx q[6],q[8];
u3(0,0,-0.98) q[8];
cx q[6],q[8];
cx q[19],q[10];
u3(0,0,-0.98) q[10];
cx q[19],q[10];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[17];
u3(0,0,-0.98) q[17];
cx q[19],q[17];
cx q[3],q[7];
u3(0,0,-0.98) q[7];
cx q[3],q[7];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[5];
u3(0,0,-0.98) q[5];
cx q[11],q[5];
cx q[2],q[11];
cx q[11],q[2];
cx q[2],q[11];
cx q[6],q[8];
cx q[8],q[6];
cx q[6],q[8];
cx q[8],q[11];
u3(0,0,-0.98) q[11];
cx q[8],q[11];
cx q[16],q[9];
u3(0,0,-0.98) q[9];
cx q[16],q[9];
cx q[10],q[17];
u3(0,0,-0.98) q[17];
cx q[10],q[17];
cx q[19],q[4];
u3(0,0,-0.98) q[4];
cx q[19],q[4];
cx q[2],q[4];
u3(0,0,-0.98) q[4];
cx q[2],q[4];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[3],q[7];
u3(0,0,-0.98) q[7];
cx q[3],q[7];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[5],q[17];
cx q[17],q[5];
cx q[5],q[17];
cx q[10],q[17];
u3(0,0,-0.98) q[17];
cx q[10],q[17];
cx q[8],q[12];
u3(0,0,-0.98) q[12];
cx q[8],q[12];
cx q[9],q[5];
cx q[5],q[9];
cx q[9],q[5];
cx q[5],q[11];
u3(0,0,-0.98) q[11];
cx q[5],q[11];
cx q[16],q[19];
u3(0,0,-0.98) q[19];
cx q[16],q[19];
cx q[8],q[12];
cx q[12],q[8];
cx q[8],q[12];
cx q[11],q[8];
u3(0,0,-0.98) q[8];
cx q[11],q[8];
cx q[12],q[15];
cx q[15],q[12];
cx q[12],q[15];
cx q[3],q[15];
u3(0,0,-0.98) q[15];
cx q[3],q[15];
cx q[16],q[2];
u3(0,0,-0.98) q[2];
cx q[16],q[2];
cx q[19],q[17];
u3(0,0,-0.98) q[17];
cx q[19],q[17];
cx q[10],q[17];
cx q[17],q[10];
cx q[10],q[17];
cx q[5],q[17];
u3(0,0,-0.98) q[17];
cx q[5],q[17];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[2],q[11];
cx q[11],q[2];
cx q[2],q[11];
cx q[11],q[8];
cx q[4],q[2];
u3(0,0,-0.98) q[2];
cx q[4],q[2];
u3(0,0,-0.98) q[8];
cx q[11],q[8];
cx q[15],q[3];
u3(0,0,-0.98) q[3];
cx q[15],q[3];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[10];
u3(0,0,-0.98) q[10];
cx q[14],q[10];
cx q[19],q[17];
u3(0,0,-0.98) q[17];
cx q[19],q[17];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[11],q[5];
u3(0,0,-0.98) q[5];
cx q[11],q[5];
cx q[14],q[3];
u3(0,0,-0.98) q[3];
cx q[14],q[3];
cx q[13],q[15];
cx q[15],q[13];
cx q[13],q[15];
cx q[13],q[10];
u3(0,0,-0.98) q[10];
cx q[13],q[10];
cx q[5],q[17];
cx q[17],q[5];
cx q[5],q[17];
cx q[9],q[5];
u3(0,0,-0.98) q[5];
cx q[9],q[5];
cx q[4],q[19];
u3(0,0,-0.98) q[19];
cx q[4],q[19];
cx q[2],q[4];
cx q[4],q[2];
cx q[2],q[4];
cx q[5],q[11];
cx q[11],q[5];
cx q[5],q[11];
cx q[2],q[11];
u3(0,0,-0.98) q[11];
cx q[2],q[11];
cx q[10],q[14];
cx q[14],q[10];
cx q[10],q[14];
cx q[3],q[14];
u3(0,0,-0.98) q[14];
cx q[3],q[14];
cx q[9],q[16];
cx q[16],q[9];
cx q[9],q[16];
cx q[10],q[17];
u3(0,0,-0.98) q[17];
cx q[10],q[17];
cx q[16],q[19];
u3(0,0,-0.98) q[19];
cx q[16],q[19];
cx q[10],q[13];
u3(0,0,-0.98) q[13];
cx q[10],q[13];
cx q[17],q[19];
cx q[19],q[17];
cx q[17],q[19];
cx q[4],q[19];
cx q[19],q[4];
cx q[4],q[19];
cx q[4],q[3];
u3(0,0,-0.98) q[3];
cx q[4],q[3];
cx q[5],q[17];
u3(0,0,-0.98) q[17];
cx q[5],q[17];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[13],q[15];
cx q[15],q[13];
cx q[13],q[15];
cx q[15],q[3];
u3(0,0,-0.98) q[3];
cx q[15],q[3];
rx(0.5) q[10];
rx(0.5) q[19];
rx(0.5) q[2];
rx(0.5) q[15];
rx(0.5) q[5];
rx(0.5) q[16];
rx(0.5) q[3];
rx(0.5) q[17];
rx(0.5) q[4];
rx(0.5) q[11];
rx(0.5) q[14];
rx(0.5) q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19];
measure q[10] -> c[0];
measure q[19] -> c[1];
measure q[2] -> c[2];
measure q[15] -> c[3];
measure q[5] -> c[4];
measure q[16] -> c[5];
measure q[3] -> c[6];
measure q[17] -> c[7];
measure q[4] -> c[8];
measure q[11] -> c[9];
measure q[14] -> c[10];
measure q[8] -> c[11];