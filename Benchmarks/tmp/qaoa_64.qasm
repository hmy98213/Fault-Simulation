// Generated from Cirq v1.0.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0, 0), q(0, 1), q(0, 2), q(0, 3), q(0, 4), q(0, 5), q(0, 6), q(0, 7), q(1, 0), q(1, 1), q(1, 2), q(1, 3), q(1, 4), q(1, 5), q(1, 6), q(1, 7), q(2, 0), q(2, 1), q(2, 2), q(2, 3), q(2, 4), q(2, 5), q(2, 6), q(2, 7), q(3, 0), q(3, 1), q(3, 2), q(3, 3), q(3, 4), q(3, 5), q(3, 6), q(3, 7), q(4, 0), q(4, 1), q(4, 2), q(4, 3), q(4, 4), q(4, 5), q(4, 6), q(4, 7), q(5, 0), q(5, 1), q(5, 2), q(5, 3), q(5, 4), q(5, 5), q(5, 6), q(5, 7), q(6, 0), q(6, 1), q(6, 2), q(6, 3), q(6, 4), q(6, 5), q(6, 6), q(6, 7), q(7, 0), q(7, 1), q(7, 2), q(7, 3), q(7, 4), q(7, 5), q(7, 6), q(7, 7)]
qreg q[64];


h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
h q[47];
h q[48];
h q[49];
h q[50];
h q[51];
h q[52];
h q[53];
h q[54];
h q[55];
h q[56];
h q[57];
h q[58];
h q[59];
h q[60];
h q[61];
h q[62];
h q[63];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[0];
rz(pi*0.495) q[1];
u3(pi*0.5,pi*1.0,pi*1.0) q[0];
u3(pi*0.5,0,pi*1.0) q[1];
sx q[0];
cx q[0],q[1];
rx(pi*0.005) q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
sxdg q[1];
s q[1];
cx q[0],q[1];
u3(pi*0.5,pi*1.505,0) q[0];
u3(pi*0.5,pi*1.505,pi*1.0) q[1];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[2];
rz(pi*0.495) q[3];
u3(pi*0.5,pi*1.0,pi*1.0) q[2];
u3(pi*0.5,0,pi*1.0) q[3];
sx q[2];
cx q[2],q[3];
rx(pi*0.005) q[2];
ry(pi*0.5) q[3];
cx q[3],q[2];
sxdg q[3];
s q[3];
cx q[2],q[3];
u3(pi*0.5,pi*1.505,0) q[2];
u3(pi*0.5,pi*1.505,pi*1.0) q[3];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[4];
rz(pi*0.495) q[5];
u3(pi*0.5,pi*1.0,pi*1.0) q[4];
u3(pi*0.5,0,pi*1.0) q[5];
sx q[4];
cx q[4],q[5];
rx(pi*0.005) q[4];
ry(pi*0.5) q[5];
cx q[5],q[4];
sxdg q[5];
s q[5];
cx q[4],q[5];
u3(pi*0.5,pi*1.505,0) q[4];
u3(pi*0.5,pi*1.505,pi*1.0) q[5];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[6];
rz(pi*0.495) q[7];
u3(pi*0.5,pi*1.0,pi*1.0) q[6];
u3(pi*0.5,0,pi*1.0) q[7];
sx q[6];
cx q[6],q[7];
rx(pi*0.005) q[6];
ry(pi*0.5) q[7];
cx q[7],q[6];
sxdg q[7];
s q[7];
cx q[6],q[7];
u3(pi*0.5,pi*1.505,0) q[6];
u3(pi*0.5,pi*1.505,pi*1.0) q[7];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[8];
rz(pi*-0.495) q[9];
u3(pi*0.5,0,0) q[8];
u3(pi*0.5,0,pi*1.0) q[9];
sx q[8];
cx q[8],q[9];
rx(pi*0.005) q[8];
ry(pi*0.5) q[9];
cx q[9],q[8];
sxdg q[9];
s q[9];
cx q[8],q[9];
u3(pi*0.5,pi*1.495,pi*1.0) q[8];
u3(pi*0.5,pi*0.495,pi*1.0) q[9];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[10];
rz(pi*-0.495) q[11];
u3(pi*0.5,0,0) q[10];
u3(pi*0.5,0,pi*1.0) q[11];
sx q[10];
cx q[10],q[11];
rx(pi*0.005) q[10];
ry(pi*0.5) q[11];
cx q[11],q[10];
sxdg q[11];
s q[11];
cx q[10],q[11];
u3(pi*0.5,pi*1.495,pi*1.0) q[10];
u3(pi*0.5,pi*0.495,pi*1.0) q[11];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[12];
rz(pi*-0.495) q[13];
u3(pi*0.5,0,0) q[12];
u3(pi*0.5,0,pi*1.0) q[13];
sx q[12];
cx q[12],q[13];
rx(pi*0.005) q[12];
ry(pi*0.5) q[13];
cx q[13],q[12];
sxdg q[13];
s q[13];
cx q[12],q[13];
u3(pi*0.5,pi*1.495,pi*1.0) q[12];
u3(pi*0.5,pi*0.495,pi*1.0) q[13];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[14];
rz(pi*0.495) q[15];
u3(pi*0.5,pi*1.0,pi*1.0) q[14];
u3(pi*0.5,0,pi*1.0) q[15];
sx q[14];
cx q[14],q[15];
rx(pi*0.005) q[14];
ry(pi*0.5) q[15];
cx q[15],q[14];
sxdg q[15];
s q[15];
cx q[14],q[15];
u3(pi*0.5,pi*1.505,0) q[14];
u3(pi*0.5,pi*1.505,pi*1.0) q[15];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[16];
rz(pi*-0.495) q[17];
u3(pi*0.5,0,0) q[16];
u3(pi*0.5,0,pi*1.0) q[17];
sx q[16];
cx q[16],q[17];
rx(pi*0.005) q[16];
ry(pi*0.5) q[17];
cx q[17],q[16];
sxdg q[17];
s q[17];
cx q[16],q[17];
u3(pi*0.5,pi*1.495,pi*1.0) q[16];
u3(pi*0.5,pi*0.495,pi*1.0) q[17];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[18];
rz(pi*0.495) q[19];
u3(pi*0.5,pi*1.0,pi*1.0) q[18];
u3(pi*0.5,0,pi*1.0) q[19];
sx q[18];
cx q[18],q[19];
rx(pi*0.005) q[18];
ry(pi*0.5) q[19];
cx q[19],q[18];
sxdg q[19];
s q[19];
cx q[18],q[19];
u3(pi*0.5,pi*1.505,0) q[18];
u3(pi*0.5,pi*1.505,pi*1.0) q[19];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[20];
rz(pi*-0.495) q[21];
u3(pi*0.5,0,0) q[20];
u3(pi*0.5,0,pi*1.0) q[21];
sx q[20];
cx q[20],q[21];
rx(pi*0.005) q[20];
ry(pi*0.5) q[21];
cx q[21],q[20];
sxdg q[21];
s q[21];
cx q[20],q[21];
u3(pi*0.5,pi*1.495,pi*1.0) q[20];
u3(pi*0.5,pi*0.495,pi*1.0) q[21];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[22];
rz(pi*0.495) q[23];
u3(pi*0.5,pi*1.0,pi*1.0) q[22];
u3(pi*0.5,0,pi*1.0) q[23];
sx q[22];
cx q[22],q[23];
rx(pi*0.005) q[22];
ry(pi*0.5) q[23];
cx q[23],q[22];
sxdg q[23];
s q[23];
cx q[22],q[23];
u3(pi*0.5,pi*1.505,0) q[22];
u3(pi*0.5,pi*1.505,pi*1.0) q[23];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[24];
rz(pi*0.495) q[25];
u3(pi*0.5,pi*1.0,pi*1.0) q[24];
u3(pi*0.5,0,pi*1.0) q[25];
sx q[24];
cx q[24],q[25];
rx(pi*0.005) q[24];
ry(pi*0.5) q[25];
cx q[25],q[24];
sxdg q[25];
s q[25];
cx q[24],q[25];
u3(pi*0.5,pi*1.505,0) q[24];
u3(pi*0.5,pi*1.505,pi*1.0) q[25];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[26];
rz(pi*-0.495) q[27];
u3(pi*0.5,0,0) q[26];
u3(pi*0.5,0,pi*1.0) q[27];
sx q[26];
cx q[26],q[27];
rx(pi*0.005) q[26];
ry(pi*0.5) q[27];
cx q[27],q[26];
sxdg q[27];
s q[27];
cx q[26],q[27];
u3(pi*0.5,pi*1.495,pi*1.0) q[26];
u3(pi*0.5,pi*0.495,pi*1.0) q[27];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[28];
rz(pi*0.495) q[29];
u3(pi*0.5,pi*1.0,pi*1.0) q[28];
u3(pi*0.5,0,pi*1.0) q[29];
sx q[28];
cx q[28],q[29];
rx(pi*0.005) q[28];
ry(pi*0.5) q[29];
cx q[29],q[28];
sxdg q[29];
s q[29];
cx q[28],q[29];
u3(pi*0.5,pi*1.505,0) q[28];
u3(pi*0.5,pi*1.505,pi*1.0) q[29];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[30];
rz(pi*-0.495) q[31];
u3(pi*0.5,0,0) q[30];
u3(pi*0.5,0,pi*1.0) q[31];
sx q[30];
cx q[30],q[31];
rx(pi*0.005) q[30];
ry(pi*0.5) q[31];
cx q[31],q[30];
sxdg q[31];
s q[31];
cx q[30],q[31];
u3(pi*0.5,pi*1.495,pi*1.0) q[30];
u3(pi*0.5,pi*0.495,pi*1.0) q[31];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[32];
rz(pi*-0.495) q[33];
u3(pi*0.5,0,0) q[32];
u3(pi*0.5,0,pi*1.0) q[33];
sx q[32];
cx q[32],q[33];
rx(pi*0.005) q[32];
ry(pi*0.5) q[33];
cx q[33],q[32];
sxdg q[33];
s q[33];
cx q[32],q[33];
u3(pi*0.5,pi*1.495,pi*1.0) q[32];
u3(pi*0.5,pi*0.495,pi*1.0) q[33];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[34];
rz(pi*-0.495) q[35];
u3(pi*0.5,0,0) q[34];
u3(pi*0.5,0,pi*1.0) q[35];
sx q[34];
cx q[34],q[35];
rx(pi*0.005) q[34];
ry(pi*0.5) q[35];
cx q[35],q[34];
sxdg q[35];
s q[35];
cx q[34],q[35];
u3(pi*0.5,pi*1.495,pi*1.0) q[34];
u3(pi*0.5,pi*0.495,pi*1.0) q[35];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[36];
rz(pi*-0.495) q[37];
u3(pi*0.5,0,0) q[36];
u3(pi*0.5,0,pi*1.0) q[37];
sx q[36];
cx q[36],q[37];
rx(pi*0.005) q[36];
ry(pi*0.5) q[37];
cx q[37],q[36];
sxdg q[37];
s q[37];
cx q[36],q[37];
u3(pi*0.5,pi*1.495,pi*1.0) q[36];
u3(pi*0.5,pi*0.495,pi*1.0) q[37];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[38];
rz(pi*-0.495) q[39];
u3(pi*0.5,0,0) q[38];
u3(pi*0.5,0,pi*1.0) q[39];
sx q[38];
cx q[38],q[39];
rx(pi*0.005) q[38];
ry(pi*0.5) q[39];
cx q[39],q[38];
sxdg q[39];
s q[39];
cx q[38],q[39];
u3(pi*0.5,pi*1.495,pi*1.0) q[38];
u3(pi*0.5,pi*0.495,pi*1.0) q[39];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[40];
rz(pi*0.495) q[41];
u3(pi*0.5,pi*1.0,pi*1.0) q[40];
u3(pi*0.5,0,pi*1.0) q[41];
sx q[40];
cx q[40],q[41];
rx(pi*0.005) q[40];
ry(pi*0.5) q[41];
cx q[41],q[40];
sxdg q[41];
s q[41];
cx q[40],q[41];
u3(pi*0.5,pi*1.505,0) q[40];
u3(pi*0.5,pi*1.505,pi*1.0) q[41];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[42];
rz(pi*-0.495) q[43];
u3(pi*0.5,0,0) q[42];
u3(pi*0.5,0,pi*1.0) q[43];
sx q[42];
cx q[42],q[43];
rx(pi*0.005) q[42];
ry(pi*0.5) q[43];
cx q[43],q[42];
sxdg q[43];
s q[43];
cx q[42],q[43];
u3(pi*0.5,pi*1.495,pi*1.0) q[42];
u3(pi*0.5,pi*0.495,pi*1.0) q[43];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[44];
rz(pi*0.495) q[45];
u3(pi*0.5,pi*1.0,pi*1.0) q[44];
u3(pi*0.5,0,pi*1.0) q[45];
sx q[44];
cx q[44],q[45];
rx(pi*0.005) q[44];
ry(pi*0.5) q[45];
cx q[45],q[44];
sxdg q[45];
s q[45];
cx q[44],q[45];
u3(pi*0.5,pi*1.505,0) q[44];
u3(pi*0.5,pi*1.505,pi*1.0) q[45];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[46];
rz(pi*0.495) q[47];
u3(pi*0.5,pi*1.0,pi*1.0) q[46];
u3(pi*0.5,0,pi*1.0) q[47];
sx q[46];
cx q[46],q[47];
rx(pi*0.005) q[46];
ry(pi*0.5) q[47];
cx q[47],q[46];
sxdg q[47];
s q[47];
cx q[46],q[47];
u3(pi*0.5,pi*1.505,0) q[46];
u3(pi*0.5,pi*1.505,pi*1.0) q[47];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[48];
rz(pi*0.495) q[49];
u3(pi*0.5,pi*1.0,pi*1.0) q[48];
u3(pi*0.5,0,pi*1.0) q[49];
sx q[48];
cx q[48],q[49];
rx(pi*0.005) q[48];
ry(pi*0.5) q[49];
cx q[49],q[48];
sxdg q[49];
s q[49];
cx q[48],q[49];
u3(pi*0.5,pi*1.505,0) q[48];
u3(pi*0.5,pi*1.505,pi*1.0) q[49];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[50];
rz(pi*-0.495) q[51];
u3(pi*0.5,0,0) q[50];
u3(pi*0.5,0,pi*1.0) q[51];
sx q[50];
cx q[50],q[51];
rx(pi*0.005) q[50];
ry(pi*0.5) q[51];
cx q[51],q[50];
sxdg q[51];
s q[51];
cx q[50],q[51];
u3(pi*0.5,pi*1.495,pi*1.0) q[50];
u3(pi*0.5,pi*0.495,pi*1.0) q[51];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[52];
rz(pi*-0.495) q[53];
u3(pi*0.5,0,0) q[52];
u3(pi*0.5,0,pi*1.0) q[53];
sx q[52];
cx q[52],q[53];
rx(pi*0.005) q[52];
ry(pi*0.5) q[53];
cx q[53],q[52];
sxdg q[53];
s q[53];
cx q[52],q[53];
u3(pi*0.5,pi*1.495,pi*1.0) q[52];
u3(pi*0.5,pi*0.495,pi*1.0) q[53];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[54];
rz(pi*0.495) q[55];
u3(pi*0.5,pi*1.0,pi*1.0) q[54];
u3(pi*0.5,0,pi*1.0) q[55];
sx q[54];
cx q[54],q[55];
rx(pi*0.005) q[54];
ry(pi*0.5) q[55];
cx q[55],q[54];
sxdg q[55];
s q[55];
cx q[54],q[55];
u3(pi*0.5,pi*1.505,0) q[54];
u3(pi*0.5,pi*1.505,pi*1.0) q[55];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[56];
rz(pi*-0.495) q[57];
u3(pi*0.5,0,0) q[56];
u3(pi*0.5,0,pi*1.0) q[57];
sx q[56];
cx q[56],q[57];
rx(pi*0.005) q[56];
ry(pi*0.5) q[57];
cx q[57],q[56];
sxdg q[57];
s q[57];
cx q[56],q[57];
u3(pi*0.5,pi*1.495,pi*1.0) q[56];
u3(pi*0.5,pi*0.495,pi*1.0) q[57];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[58];
rz(pi*0.495) q[59];
u3(pi*0.5,pi*1.0,pi*1.0) q[58];
u3(pi*0.5,0,pi*1.0) q[59];
sx q[58];
cx q[58],q[59];
rx(pi*0.005) q[58];
ry(pi*0.5) q[59];
cx q[59],q[58];
sxdg q[59];
s q[59];
cx q[58],q[59];
u3(pi*0.5,pi*1.505,0) q[58];
u3(pi*0.5,pi*1.505,pi*1.0) q[59];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[60];
rz(pi*-0.495) q[61];
u3(pi*0.5,0,0) q[60];
u3(pi*0.5,0,pi*1.0) q[61];
sx q[60];
cx q[60],q[61];
rx(pi*0.005) q[60];
ry(pi*0.5) q[61];
cx q[61],q[60];
sxdg q[61];
s q[61];
cx q[60],q[61];
u3(pi*0.5,pi*1.495,pi*1.0) q[60];
u3(pi*0.5,pi*0.495,pi*1.0) q[61];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[62];
rz(pi*-0.495) q[63];
u3(pi*0.5,0,0) q[62];
u3(pi*0.5,0,pi*1.0) q[63];
sx q[62];
cx q[62],q[63];
rx(pi*0.005) q[62];
ry(pi*0.5) q[63];
cx q[63],q[62];
sxdg q[63];
s q[63];
cx q[62],q[63];
u3(pi*0.5,pi*1.495,pi*1.0) q[62];
u3(pi*0.5,pi*0.495,pi*1.0) q[63];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[1];
rz(pi*-0.495) q[2];
u3(pi*0.5,0,0) q[1];
u3(pi*0.5,0,pi*1.0) q[2];
sx q[1];
cx q[1],q[2];
rx(pi*0.005) q[1];
ry(pi*0.5) q[2];
cx q[2],q[1];
sxdg q[2];
s q[2];
cx q[1],q[2];
u3(pi*0.5,pi*1.495,pi*1.0) q[1];
u3(pi*0.5,pi*0.495,pi*1.0) q[2];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[3];
rz(pi*0.495) q[4];
u3(pi*0.5,pi*1.0,pi*1.0) q[3];
u3(pi*0.5,0,pi*1.0) q[4];
sx q[3];
cx q[3],q[4];
rx(pi*0.005) q[3];
ry(pi*0.5) q[4];
cx q[4],q[3];
sxdg q[4];
s q[4];
cx q[3],q[4];
u3(pi*0.5,pi*1.505,0) q[3];
u3(pi*0.5,pi*1.505,pi*1.0) q[4];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[5];
rz(pi*-0.495) q[6];
u3(pi*0.5,0,0) q[5];
u3(pi*0.5,0,pi*1.0) q[6];
sx q[5];
cx q[5],q[6];
rx(pi*0.005) q[5];
ry(pi*0.5) q[6];
cx q[6],q[5];
sxdg q[6];
s q[6];
cx q[5],q[6];
u3(pi*0.5,pi*1.495,pi*1.0) q[5];
u3(pi*0.5,pi*0.495,pi*1.0) q[6];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[9];
rz(pi*-0.495) q[10];
u3(pi*0.5,0,0) q[9];
u3(pi*0.5,0,pi*1.0) q[10];
sx q[9];
cx q[9],q[10];
rx(pi*0.005) q[9];
ry(pi*0.5) q[10];
cx q[10],q[9];
sxdg q[10];
s q[10];
cx q[9],q[10];
u3(pi*0.5,pi*1.495,pi*1.0) q[9];
u3(pi*0.5,pi*0.495,pi*1.0) q[10];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[11];
rz(pi*0.495) q[12];
u3(pi*0.5,pi*1.0,pi*1.0) q[11];
u3(pi*0.5,0,pi*1.0) q[12];
sx q[11];
cx q[11],q[12];
rx(pi*0.005) q[11];
ry(pi*0.5) q[12];
cx q[12],q[11];
sxdg q[12];
s q[12];
cx q[11],q[12];
u3(pi*0.5,pi*1.505,0) q[11];
u3(pi*0.5,pi*1.505,pi*1.0) q[12];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[13];
rz(pi*0.495) q[14];
u3(pi*0.5,pi*1.0,pi*1.0) q[13];
u3(pi*0.5,0,pi*1.0) q[14];
sx q[13];
cx q[13],q[14];
rx(pi*0.005) q[13];
ry(pi*0.5) q[14];
cx q[14],q[13];
sxdg q[14];
s q[14];
cx q[13],q[14];
u3(pi*0.5,pi*1.505,0) q[13];
u3(pi*0.5,pi*1.505,pi*1.0) q[14];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[17];
rz(pi*-0.495) q[18];
u3(pi*0.5,0,0) q[17];
u3(pi*0.5,0,pi*1.0) q[18];
sx q[17];
cx q[17],q[18];
rx(pi*0.005) q[17];
ry(pi*0.5) q[18];
cx q[18],q[17];
sxdg q[18];
s q[18];
cx q[17],q[18];
u3(pi*0.5,pi*1.495,pi*1.0) q[17];
u3(pi*0.5,pi*0.495,pi*1.0) q[18];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[19];
rz(pi*0.495) q[20];
u3(pi*0.5,pi*1.0,pi*1.0) q[19];
u3(pi*0.5,0,pi*1.0) q[20];
sx q[19];
cx q[19],q[20];
rx(pi*0.005) q[19];
ry(pi*0.5) q[20];
cx q[20],q[19];
sxdg q[20];
s q[20];
cx q[19],q[20];
u3(pi*0.5,pi*1.505,0) q[19];
u3(pi*0.5,pi*1.505,pi*1.0) q[20];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[21];
rz(pi*0.495) q[22];
u3(pi*0.5,pi*1.0,pi*1.0) q[21];
u3(pi*0.5,0,pi*1.0) q[22];
sx q[21];
cx q[21],q[22];
rx(pi*0.005) q[21];
ry(pi*0.5) q[22];
cx q[22],q[21];
sxdg q[22];
s q[22];
cx q[21],q[22];
u3(pi*0.5,pi*1.505,0) q[21];
u3(pi*0.5,pi*1.505,pi*1.0) q[22];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[25];
rz(pi*-0.495) q[26];
u3(pi*0.5,0,0) q[25];
u3(pi*0.5,0,pi*1.0) q[26];
sx q[25];
cx q[25],q[26];
rx(pi*0.005) q[25];
ry(pi*0.5) q[26];
cx q[26],q[25];
sxdg q[26];
s q[26];
cx q[25],q[26];
u3(pi*0.5,pi*1.495,pi*1.0) q[25];
u3(pi*0.5,pi*0.495,pi*1.0) q[26];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[27];
rz(pi*0.495) q[28];
u3(pi*0.5,pi*1.0,pi*1.0) q[27];
u3(pi*0.5,0,pi*1.0) q[28];
sx q[27];
cx q[27],q[28];
rx(pi*0.005) q[27];
ry(pi*0.5) q[28];
cx q[28],q[27];
sxdg q[28];
s q[28];
cx q[27],q[28];
u3(pi*0.5,pi*1.505,0) q[27];
u3(pi*0.5,pi*1.505,pi*1.0) q[28];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[29];
rz(pi*0.495) q[30];
u3(pi*0.5,pi*1.0,pi*1.0) q[29];
u3(pi*0.5,0,pi*1.0) q[30];
sx q[29];
cx q[29],q[30];
rx(pi*0.005) q[29];
ry(pi*0.5) q[30];
cx q[30],q[29];
sxdg q[30];
s q[30];
cx q[29],q[30];
u3(pi*0.5,pi*1.505,0) q[29];
u3(pi*0.5,pi*1.505,pi*1.0) q[30];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[33];
rz(pi*-0.495) q[34];
u3(pi*0.5,0,0) q[33];
u3(pi*0.5,0,pi*1.0) q[34];
sx q[33];
cx q[33],q[34];
rx(pi*0.005) q[33];
ry(pi*0.5) q[34];
cx q[34],q[33];
sxdg q[34];
s q[34];
cx q[33],q[34];
u3(pi*0.5,pi*1.495,pi*1.0) q[33];
u3(pi*0.5,pi*0.495,pi*1.0) q[34];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[35];
rz(pi*-0.495) q[36];
u3(pi*0.5,0,0) q[35];
u3(pi*0.5,0,pi*1.0) q[36];
sx q[35];
cx q[35],q[36];
rx(pi*0.005) q[35];
ry(pi*0.5) q[36];
cx q[36],q[35];
sxdg q[36];
s q[36];
cx q[35],q[36];
u3(pi*0.5,pi*1.495,pi*1.0) q[35];
u3(pi*0.5,pi*0.495,pi*1.0) q[36];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[37];
rz(pi*0.495) q[38];
u3(pi*0.5,pi*1.0,pi*1.0) q[37];
u3(pi*0.5,0,pi*1.0) q[38];
sx q[37];
cx q[37],q[38];
rx(pi*0.005) q[37];
ry(pi*0.5) q[38];
cx q[38],q[37];
sxdg q[38];
s q[38];
cx q[37],q[38];
u3(pi*0.5,pi*1.505,0) q[37];
u3(pi*0.5,pi*1.505,pi*1.0) q[38];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[41];
rz(pi*0.495) q[42];
u3(pi*0.5,pi*1.0,pi*1.0) q[41];
u3(pi*0.5,0,pi*1.0) q[42];
sx q[41];
cx q[41],q[42];
rx(pi*0.005) q[41];
ry(pi*0.5) q[42];
cx q[42],q[41];
sxdg q[42];
s q[42];
cx q[41],q[42];
u3(pi*0.5,pi*1.505,0) q[41];
u3(pi*0.5,pi*1.505,pi*1.0) q[42];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[43];
rz(pi*-0.495) q[44];
u3(pi*0.5,0,0) q[43];
u3(pi*0.5,0,pi*1.0) q[44];
sx q[43];
cx q[43],q[44];
rx(pi*0.005) q[43];
ry(pi*0.5) q[44];
cx q[44],q[43];
sxdg q[44];
s q[44];
cx q[43],q[44];
u3(pi*0.5,pi*1.495,pi*1.0) q[43];
u3(pi*0.5,pi*0.495,pi*1.0) q[44];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[45];
rz(pi*0.495) q[46];
u3(pi*0.5,pi*1.0,pi*1.0) q[45];
u3(pi*0.5,0,pi*1.0) q[46];
sx q[45];
cx q[45],q[46];
rx(pi*0.005) q[45];
ry(pi*0.5) q[46];
cx q[46],q[45];
sxdg q[46];
s q[46];
cx q[45],q[46];
u3(pi*0.5,pi*1.505,0) q[45];
u3(pi*0.5,pi*1.505,pi*1.0) q[46];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[49];
rz(pi*-0.495) q[50];
u3(pi*0.5,0,0) q[49];
u3(pi*0.5,0,pi*1.0) q[50];
sx q[49];
cx q[49],q[50];
rx(pi*0.005) q[49];
ry(pi*0.5) q[50];
cx q[50],q[49];
sxdg q[50];
s q[50];
cx q[49],q[50];
u3(pi*0.5,pi*1.495,pi*1.0) q[49];
u3(pi*0.5,pi*0.495,pi*1.0) q[50];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[51];
rz(pi*0.495) q[52];
u3(pi*0.5,pi*1.0,pi*1.0) q[51];
u3(pi*0.5,0,pi*1.0) q[52];
sx q[51];
cx q[51],q[52];
rx(pi*0.005) q[51];
ry(pi*0.5) q[52];
cx q[52],q[51];
sxdg q[52];
s q[52];
cx q[51],q[52];
u3(pi*0.5,pi*1.505,0) q[51];
u3(pi*0.5,pi*1.505,pi*1.0) q[52];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[53];
rz(pi*-0.495) q[54];
u3(pi*0.5,0,0) q[53];
u3(pi*0.5,0,pi*1.0) q[54];
sx q[53];
cx q[53],q[54];
rx(pi*0.005) q[53];
ry(pi*0.5) q[54];
cx q[54],q[53];
sxdg q[54];
s q[54];
cx q[53],q[54];
u3(pi*0.5,pi*1.495,pi*1.0) q[53];
u3(pi*0.5,pi*0.495,pi*1.0) q[54];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[57];
rz(pi*-0.495) q[58];
u3(pi*0.5,0,0) q[57];
u3(pi*0.5,0,pi*1.0) q[58];
sx q[57];
cx q[57],q[58];
rx(pi*0.005) q[57];
ry(pi*0.5) q[58];
cx q[58],q[57];
sxdg q[58];
s q[58];
cx q[57],q[58];
u3(pi*0.5,pi*1.495,pi*1.0) q[57];
u3(pi*0.5,pi*0.495,pi*1.0) q[58];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[59];
rz(pi*0.495) q[60];
u3(pi*0.5,pi*1.0,pi*1.0) q[59];
u3(pi*0.5,0,pi*1.0) q[60];
sx q[59];
cx q[59],q[60];
rx(pi*0.005) q[59];
ry(pi*0.5) q[60];
cx q[60],q[59];
sxdg q[60];
s q[60];
cx q[59],q[60];
u3(pi*0.5,pi*1.505,0) q[59];
u3(pi*0.5,pi*1.505,pi*1.0) q[60];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[61];
rz(pi*0.495) q[62];
u3(pi*0.5,pi*1.0,pi*1.0) q[61];
u3(pi*0.5,0,pi*1.0) q[62];
sx q[61];
cx q[61],q[62];
rx(pi*0.005) q[61];
ry(pi*0.5) q[62];
cx q[62],q[61];
sxdg q[62];
s q[62];
cx q[61],q[62];
u3(pi*0.5,pi*1.505,0) q[61];
u3(pi*0.5,pi*1.505,pi*1.0) q[62];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[0];
rz(pi*-0.495) q[8];
u3(pi*0.5,0,0) q[0];
u3(pi*0.5,0,pi*1.0) q[8];
sx q[0];
cx q[0],q[8];
rx(pi*0.005) q[0];
ry(pi*0.5) q[8];
cx q[8],q[0];
sxdg q[8];
s q[8];
cx q[0],q[8];
u3(pi*0.5,pi*1.495,pi*1.0) q[0];
u3(pi*0.5,pi*0.495,pi*1.0) q[8];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[1];
rz(pi*0.495) q[9];
u3(pi*0.5,pi*1.0,pi*1.0) q[1];
u3(pi*0.5,0,pi*1.0) q[9];
sx q[1];
cx q[1],q[9];
rx(pi*0.005) q[1];
ry(pi*0.5) q[9];
cx q[9],q[1];
sxdg q[9];
s q[9];
cx q[1],q[9];
u3(pi*0.5,pi*1.505,0) q[1];
u3(pi*0.5,pi*1.505,pi*1.0) q[9];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[2];
rz(pi*0.495) q[10];
u3(pi*0.5,pi*1.0,pi*1.0) q[2];
u3(pi*0.5,0,pi*1.0) q[10];
sx q[2];
cx q[2],q[10];
rx(pi*0.005) q[2];
ry(pi*0.5) q[10];
cx q[10],q[2];
sxdg q[10];
s q[10];
cx q[2],q[10];
u3(pi*0.5,pi*1.505,0) q[2];
u3(pi*0.5,pi*1.505,pi*1.0) q[10];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[3];
rz(pi*0.495) q[11];
u3(pi*0.5,pi*1.0,pi*1.0) q[3];
u3(pi*0.5,0,pi*1.0) q[11];
sx q[3];
cx q[3],q[11];
rx(pi*0.005) q[3];
ry(pi*0.5) q[11];
cx q[11],q[3];
sxdg q[11];
s q[11];
cx q[3],q[11];
u3(pi*0.5,pi*1.505,0) q[3];
u3(pi*0.5,pi*1.505,pi*1.0) q[11];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[4];
rz(pi*0.495) q[12];
u3(pi*0.5,pi*1.0,pi*1.0) q[4];
u3(pi*0.5,0,pi*1.0) q[12];
sx q[4];
cx q[4],q[12];
rx(pi*0.005) q[4];
ry(pi*0.5) q[12];
cx q[12],q[4];
sxdg q[12];
s q[12];
cx q[4],q[12];
u3(pi*0.5,pi*1.505,0) q[4];
u3(pi*0.5,pi*1.505,pi*1.0) q[12];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[5];
rz(pi*0.495) q[13];
u3(pi*0.5,pi*1.0,pi*1.0) q[5];
u3(pi*0.5,0,pi*1.0) q[13];
sx q[5];
cx q[5],q[13];
rx(pi*0.005) q[5];
ry(pi*0.5) q[13];
cx q[13],q[5];
sxdg q[13];
s q[13];
cx q[5],q[13];
u3(pi*0.5,pi*1.505,0) q[5];
u3(pi*0.5,pi*1.505,pi*1.0) q[13];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[6];
rz(pi*-0.495) q[14];
u3(pi*0.5,0,0) q[6];
u3(pi*0.5,0,pi*1.0) q[14];
sx q[6];
cx q[6],q[14];
rx(pi*0.005) q[6];
ry(pi*0.5) q[14];
cx q[14],q[6];
sxdg q[14];
s q[14];
cx q[6],q[14];
u3(pi*0.5,pi*1.495,pi*1.0) q[6];
u3(pi*0.5,pi*0.495,pi*1.0) q[14];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[7];
rz(pi*-0.495) q[15];
u3(pi*0.5,0,0) q[7];
u3(pi*0.5,0,pi*1.0) q[15];
sx q[7];
cx q[7],q[15];
rx(pi*0.005) q[7];
ry(pi*0.5) q[15];
cx q[15],q[7];
sxdg q[15];
s q[15];
cx q[7],q[15];
u3(pi*0.5,pi*1.495,pi*1.0) q[7];
u3(pi*0.5,pi*0.495,pi*1.0) q[15];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[16];
rz(pi*0.495) q[24];
u3(pi*0.5,pi*1.0,pi*1.0) q[16];
u3(pi*0.5,0,pi*1.0) q[24];
sx q[16];
cx q[16],q[24];
rx(pi*0.005) q[16];
ry(pi*0.5) q[24];
cx q[24],q[16];
sxdg q[24];
s q[24];
cx q[16],q[24];
u3(pi*0.5,pi*1.505,0) q[16];
u3(pi*0.5,pi*1.505,pi*1.0) q[24];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[17];
rz(pi*0.495) q[25];
u3(pi*0.5,pi*1.0,pi*1.0) q[17];
u3(pi*0.5,0,pi*1.0) q[25];
sx q[17];
cx q[17],q[25];
rx(pi*0.005) q[17];
ry(pi*0.5) q[25];
cx q[25],q[17];
sxdg q[25];
s q[25];
cx q[17],q[25];
u3(pi*0.5,pi*1.505,0) q[17];
u3(pi*0.5,pi*1.505,pi*1.0) q[25];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[18];
rz(pi*0.495) q[26];
u3(pi*0.5,pi*1.0,pi*1.0) q[18];
u3(pi*0.5,0,pi*1.0) q[26];
sx q[18];
cx q[18],q[26];
rx(pi*0.005) q[18];
ry(pi*0.5) q[26];
cx q[26],q[18];
sxdg q[26];
s q[26];
cx q[18],q[26];
u3(pi*0.5,pi*1.505,0) q[18];
u3(pi*0.5,pi*1.505,pi*1.0) q[26];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[19];
rz(pi*-0.495) q[27];
u3(pi*0.5,0,0) q[19];
u3(pi*0.5,0,pi*1.0) q[27];
sx q[19];
cx q[19],q[27];
rx(pi*0.005) q[19];
ry(pi*0.5) q[27];
cx q[27],q[19];
sxdg q[27];
s q[27];
cx q[19],q[27];
u3(pi*0.5,pi*1.495,pi*1.0) q[19];
u3(pi*0.5,pi*0.495,pi*1.0) q[27];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[20];
rz(pi*0.495) q[28];
u3(pi*0.5,pi*1.0,pi*1.0) q[20];
u3(pi*0.5,0,pi*1.0) q[28];
sx q[20];
cx q[20],q[28];
rx(pi*0.005) q[20];
ry(pi*0.5) q[28];
cx q[28],q[20];
sxdg q[28];
s q[28];
cx q[20],q[28];
u3(pi*0.5,pi*1.505,0) q[20];
u3(pi*0.5,pi*1.505,pi*1.0) q[28];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[21];
rz(pi*-0.495) q[29];
u3(pi*0.5,0,0) q[21];
u3(pi*0.5,0,pi*1.0) q[29];
sx q[21];
cx q[21],q[29];
rx(pi*0.005) q[21];
ry(pi*0.5) q[29];
cx q[29],q[21];
sxdg q[29];
s q[29];
cx q[21],q[29];
u3(pi*0.5,pi*1.495,pi*1.0) q[21];
u3(pi*0.5,pi*0.495,pi*1.0) q[29];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[22];
rz(pi*-0.495) q[30];
u3(pi*0.5,0,0) q[22];
u3(pi*0.5,0,pi*1.0) q[30];
sx q[22];
cx q[22],q[30];
rx(pi*0.005) q[22];
ry(pi*0.5) q[30];
cx q[30],q[22];
sxdg q[30];
s q[30];
cx q[22],q[30];
u3(pi*0.5,pi*1.495,pi*1.0) q[22];
u3(pi*0.5,pi*0.495,pi*1.0) q[30];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[23];
rz(pi*0.495) q[31];
u3(pi*0.5,pi*1.0,pi*1.0) q[23];
u3(pi*0.5,0,pi*1.0) q[31];
sx q[23];
cx q[23],q[31];
rx(pi*0.005) q[23];
ry(pi*0.5) q[31];
cx q[31],q[23];
sxdg q[31];
s q[31];
cx q[23],q[31];
u3(pi*0.5,pi*1.505,0) q[23];
u3(pi*0.5,pi*1.505,pi*1.0) q[31];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[32];
rz(pi*0.495) q[40];
u3(pi*0.5,pi*1.0,pi*1.0) q[32];
u3(pi*0.5,0,pi*1.0) q[40];
sx q[32];
cx q[32],q[40];
rx(pi*0.005) q[32];
ry(pi*0.5) q[40];
cx q[40],q[32];
sxdg q[40];
s q[40];
cx q[32],q[40];
u3(pi*0.5,pi*1.505,0) q[32];
u3(pi*0.5,pi*1.505,pi*1.0) q[40];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[33];
rz(pi*0.495) q[41];
u3(pi*0.5,pi*1.0,pi*1.0) q[33];
u3(pi*0.5,0,pi*1.0) q[41];
sx q[33];
cx q[33],q[41];
rx(pi*0.005) q[33];
ry(pi*0.5) q[41];
cx q[41],q[33];
sxdg q[41];
s q[41];
cx q[33],q[41];
u3(pi*0.5,pi*1.505,0) q[33];
u3(pi*0.5,pi*1.505,pi*1.0) q[41];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[34];
rz(pi*0.495) q[42];
u3(pi*0.5,pi*1.0,pi*1.0) q[34];
u3(pi*0.5,0,pi*1.0) q[42];
sx q[34];
cx q[34],q[42];
rx(pi*0.005) q[34];
ry(pi*0.5) q[42];
cx q[42],q[34];
sxdg q[42];
s q[42];
cx q[34],q[42];
u3(pi*0.5,pi*1.505,0) q[34];
u3(pi*0.5,pi*1.505,pi*1.0) q[42];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[35];
rz(pi*-0.495) q[43];
u3(pi*0.5,0,0) q[35];
u3(pi*0.5,0,pi*1.0) q[43];
sx q[35];
cx q[35],q[43];
rx(pi*0.005) q[35];
ry(pi*0.5) q[43];
cx q[43],q[35];
sxdg q[43];
s q[43];
cx q[35],q[43];
u3(pi*0.5,pi*1.495,pi*1.0) q[35];
u3(pi*0.5,pi*0.495,pi*1.0) q[43];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[36];
rz(pi*-0.495) q[44];
u3(pi*0.5,0,0) q[36];
u3(pi*0.5,0,pi*1.0) q[44];
sx q[36];
cx q[36],q[44];
rx(pi*0.005) q[36];
ry(pi*0.5) q[44];
cx q[44],q[36];
sxdg q[44];
s q[44];
cx q[36],q[44];
u3(pi*0.5,pi*1.495,pi*1.0) q[36];
u3(pi*0.5,pi*0.495,pi*1.0) q[44];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[37];
rz(pi*0.495) q[45];
u3(pi*0.5,pi*1.0,pi*1.0) q[37];
u3(pi*0.5,0,pi*1.0) q[45];
sx q[37];
cx q[37],q[45];
rx(pi*0.005) q[37];
ry(pi*0.5) q[45];
cx q[45],q[37];
sxdg q[45];
s q[45];
cx q[37],q[45];
u3(pi*0.5,pi*1.505,0) q[37];
u3(pi*0.5,pi*1.505,pi*1.0) q[45];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[38];
rz(pi*-0.495) q[46];
u3(pi*0.5,0,0) q[38];
u3(pi*0.5,0,pi*1.0) q[46];
sx q[38];
cx q[38],q[46];
rx(pi*0.005) q[38];
ry(pi*0.5) q[46];
cx q[46],q[38];
sxdg q[46];
s q[46];
cx q[38],q[46];
u3(pi*0.5,pi*1.495,pi*1.0) q[38];
u3(pi*0.5,pi*0.495,pi*1.0) q[46];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[39];
rz(pi*-0.495) q[47];
u3(pi*0.5,0,0) q[39];
u3(pi*0.5,0,pi*1.0) q[47];
sx q[39];
cx q[39],q[47];
rx(pi*0.005) q[39];
ry(pi*0.5) q[47];
cx q[47],q[39];
sxdg q[47];
s q[47];
cx q[39],q[47];
u3(pi*0.5,pi*1.495,pi*1.0) q[39];
u3(pi*0.5,pi*0.495,pi*1.0) q[47];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[48];
rz(pi*0.495) q[56];
u3(pi*0.5,pi*1.0,pi*1.0) q[48];
u3(pi*0.5,0,pi*1.0) q[56];
sx q[48];
cx q[48],q[56];
rx(pi*0.005) q[48];
ry(pi*0.5) q[56];
cx q[56],q[48];
sxdg q[56];
s q[56];
cx q[48],q[56];
u3(pi*0.5,pi*1.505,0) q[48];
u3(pi*0.5,pi*1.505,pi*1.0) q[56];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[49];
rz(pi*-0.495) q[57];
u3(pi*0.5,0,0) q[49];
u3(pi*0.5,0,pi*1.0) q[57];
sx q[49];
cx q[49],q[57];
rx(pi*0.005) q[49];
ry(pi*0.5) q[57];
cx q[57],q[49];
sxdg q[57];
s q[57];
cx q[49],q[57];
u3(pi*0.5,pi*1.495,pi*1.0) q[49];
u3(pi*0.5,pi*0.495,pi*1.0) q[57];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[50];
rz(pi*0.495) q[58];
u3(pi*0.5,pi*1.0,pi*1.0) q[50];
u3(pi*0.5,0,pi*1.0) q[58];
sx q[50];
cx q[50],q[58];
rx(pi*0.005) q[50];
ry(pi*0.5) q[58];
cx q[58],q[50];
sxdg q[58];
s q[58];
cx q[50],q[58];
u3(pi*0.5,pi*1.505,0) q[50];
u3(pi*0.5,pi*1.505,pi*1.0) q[58];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[51];
rz(pi*-0.495) q[59];
u3(pi*0.5,0,0) q[51];
u3(pi*0.5,0,pi*1.0) q[59];
sx q[51];
cx q[51],q[59];
rx(pi*0.005) q[51];
ry(pi*0.5) q[59];
cx q[59],q[51];
sxdg q[59];
s q[59];
cx q[51],q[59];
u3(pi*0.5,pi*1.495,pi*1.0) q[51];
u3(pi*0.5,pi*0.495,pi*1.0) q[59];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[52];
rz(pi*0.495) q[60];
u3(pi*0.5,pi*1.0,pi*1.0) q[52];
u3(pi*0.5,0,pi*1.0) q[60];
sx q[52];
cx q[52],q[60];
rx(pi*0.005) q[52];
ry(pi*0.5) q[60];
cx q[60],q[52];
sxdg q[60];
s q[60];
cx q[52],q[60];
u3(pi*0.5,pi*1.505,0) q[52];
u3(pi*0.5,pi*1.505,pi*1.0) q[60];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[53];
rz(pi*0.495) q[61];
u3(pi*0.5,pi*1.0,pi*1.0) q[53];
u3(pi*0.5,0,pi*1.0) q[61];
sx q[53];
cx q[53],q[61];
rx(pi*0.005) q[53];
ry(pi*0.5) q[61];
cx q[61],q[53];
sxdg q[61];
s q[61];
cx q[53],q[61];
u3(pi*0.5,pi*1.505,0) q[53];
u3(pi*0.5,pi*1.505,pi*1.0) q[61];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[54];
rz(pi*-0.495) q[62];
u3(pi*0.5,0,0) q[54];
u3(pi*0.5,0,pi*1.0) q[62];
sx q[54];
cx q[54],q[62];
rx(pi*0.005) q[54];
ry(pi*0.5) q[62];
cx q[62],q[54];
sxdg q[62];
s q[62];
cx q[54],q[62];
u3(pi*0.5,pi*1.495,pi*1.0) q[54];
u3(pi*0.5,pi*0.495,pi*1.0) q[62];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[55];
rz(pi*-0.495) q[63];
u3(pi*0.5,0,0) q[55];
u3(pi*0.5,0,pi*1.0) q[63];
sx q[55];
cx q[55],q[63];
rx(pi*0.005) q[55];
ry(pi*0.5) q[63];
cx q[63],q[55];
sxdg q[63];
s q[63];
cx q[55],q[63];
u3(pi*0.5,pi*1.495,pi*1.0) q[55];
u3(pi*0.5,pi*0.495,pi*1.0) q[63];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[8];
rz(pi*-0.495) q[16];
u3(pi*0.5,0,0) q[8];
u3(pi*0.5,0,pi*1.0) q[16];
sx q[8];
cx q[8],q[16];
rx(pi*0.005) q[8];
ry(pi*0.5) q[16];
cx q[16],q[8];
sxdg q[16];
s q[16];
cx q[8],q[16];
u3(pi*0.5,pi*1.495,pi*1.0) q[8];
u3(pi*0.5,pi*0.495,pi*1.0) q[16];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[9];
rz(pi*-0.495) q[17];
u3(pi*0.5,0,0) q[9];
u3(pi*0.5,0,pi*1.0) q[17];
sx q[9];
cx q[9],q[17];
rx(pi*0.005) q[9];
ry(pi*0.5) q[17];
cx q[17],q[9];
sxdg q[17];
s q[17];
cx q[9],q[17];
u3(pi*0.5,pi*1.495,pi*1.0) q[9];
u3(pi*0.5,pi*0.495,pi*1.0) q[17];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[10];
rz(pi*0.495) q[18];
u3(pi*0.5,pi*1.0,pi*1.0) q[10];
u3(pi*0.5,0,pi*1.0) q[18];
sx q[10];
cx q[10],q[18];
rx(pi*0.005) q[10];
ry(pi*0.5) q[18];
cx q[18],q[10];
sxdg q[18];
s q[18];
cx q[10],q[18];
u3(pi*0.5,pi*1.505,0) q[10];
u3(pi*0.5,pi*1.505,pi*1.0) q[18];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[11];
rz(pi*0.495) q[19];
u3(pi*0.5,pi*1.0,pi*1.0) q[11];
u3(pi*0.5,0,pi*1.0) q[19];
sx q[11];
cx q[11],q[19];
rx(pi*0.005) q[11];
ry(pi*0.5) q[19];
cx q[19],q[11];
sxdg q[19];
s q[19];
cx q[11],q[19];
u3(pi*0.5,pi*1.505,0) q[11];
u3(pi*0.5,pi*1.505,pi*1.0) q[19];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[12];
rz(pi*-0.495) q[20];
u3(pi*0.5,0,0) q[12];
u3(pi*0.5,0,pi*1.0) q[20];
sx q[12];
cx q[12],q[20];
rx(pi*0.005) q[12];
ry(pi*0.5) q[20];
cx q[20],q[12];
sxdg q[20];
s q[20];
cx q[12],q[20];
u3(pi*0.5,pi*1.495,pi*1.0) q[12];
u3(pi*0.5,pi*0.495,pi*1.0) q[20];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[13];
rz(pi*0.495) q[21];
u3(pi*0.5,pi*1.0,pi*1.0) q[13];
u3(pi*0.5,0,pi*1.0) q[21];
sx q[13];
cx q[13],q[21];
rx(pi*0.005) q[13];
ry(pi*0.5) q[21];
cx q[21],q[13];
sxdg q[21];
s q[21];
cx q[13],q[21];
u3(pi*0.5,pi*1.505,0) q[13];
u3(pi*0.5,pi*1.505,pi*1.0) q[21];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[14];
rz(pi*0.495) q[22];
u3(pi*0.5,pi*1.0,pi*1.0) q[14];
u3(pi*0.5,0,pi*1.0) q[22];
sx q[14];
cx q[14],q[22];
rx(pi*0.005) q[14];
ry(pi*0.5) q[22];
cx q[22],q[14];
sxdg q[22];
s q[22];
cx q[14],q[22];
u3(pi*0.5,pi*1.505,0) q[14];
u3(pi*0.5,pi*1.505,pi*1.0) q[22];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[15];
rz(pi*-0.495) q[23];
u3(pi*0.5,0,0) q[15];
u3(pi*0.5,0,pi*1.0) q[23];
sx q[15];
cx q[15],q[23];
rx(pi*0.005) q[15];
ry(pi*0.5) q[23];
cx q[23],q[15];
sxdg q[23];
s q[23];
cx q[15],q[23];
u3(pi*0.5,pi*1.495,pi*1.0) q[15];
u3(pi*0.5,pi*0.495,pi*1.0) q[23];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[24];
rz(pi*0.495) q[32];
u3(pi*0.5,pi*1.0,pi*1.0) q[24];
u3(pi*0.5,0,pi*1.0) q[32];
sx q[24];
cx q[24],q[32];
rx(pi*0.005) q[24];
ry(pi*0.5) q[32];
cx q[32],q[24];
sxdg q[32];
s q[32];
cx q[24],q[32];
u3(pi*0.5,pi*1.505,0) q[24];
u3(pi*0.5,pi*1.505,pi*1.0) q[32];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[25];
rz(pi*0.495) q[33];
u3(pi*0.5,pi*1.0,pi*1.0) q[25];
u3(pi*0.5,0,pi*1.0) q[33];
sx q[25];
cx q[25],q[33];
rx(pi*0.005) q[25];
ry(pi*0.5) q[33];
cx q[33],q[25];
sxdg q[33];
s q[33];
cx q[25],q[33];
u3(pi*0.5,pi*1.505,0) q[25];
u3(pi*0.5,pi*1.505,pi*1.0) q[33];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[26];
rz(pi*0.495) q[34];
u3(pi*0.5,pi*1.0,pi*1.0) q[26];
u3(pi*0.5,0,pi*1.0) q[34];
sx q[26];
cx q[26],q[34];
rx(pi*0.005) q[26];
ry(pi*0.5) q[34];
cx q[34],q[26];
sxdg q[34];
s q[34];
cx q[26],q[34];
u3(pi*0.5,pi*1.505,0) q[26];
u3(pi*0.5,pi*1.505,pi*1.0) q[34];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[27];
rz(pi*0.495) q[35];
u3(pi*0.5,pi*1.0,pi*1.0) q[27];
u3(pi*0.5,0,pi*1.0) q[35];
sx q[27];
cx q[27],q[35];
rx(pi*0.005) q[27];
ry(pi*0.5) q[35];
cx q[35],q[27];
sxdg q[35];
s q[35];
cx q[27],q[35];
u3(pi*0.5,pi*1.505,0) q[27];
u3(pi*0.5,pi*1.505,pi*1.0) q[35];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[28];
rz(pi*0.495) q[36];
u3(pi*0.5,pi*1.0,pi*1.0) q[28];
u3(pi*0.5,0,pi*1.0) q[36];
sx q[28];
cx q[28],q[36];
rx(pi*0.005) q[28];
ry(pi*0.5) q[36];
cx q[36],q[28];
sxdg q[36];
s q[36];
cx q[28],q[36];
u3(pi*0.5,pi*1.505,0) q[28];
u3(pi*0.5,pi*1.505,pi*1.0) q[36];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[29];
rz(pi*-0.495) q[37];
u3(pi*0.5,0,0) q[29];
u3(pi*0.5,0,pi*1.0) q[37];
sx q[29];
cx q[29],q[37];
rx(pi*0.005) q[29];
ry(pi*0.5) q[37];
cx q[37],q[29];
sxdg q[37];
s q[37];
cx q[29],q[37];
u3(pi*0.5,pi*1.495,pi*1.0) q[29];
u3(pi*0.5,pi*0.495,pi*1.0) q[37];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[30];
rz(pi*-0.495) q[38];
u3(pi*0.5,0,0) q[30];
u3(pi*0.5,0,pi*1.0) q[38];
sx q[30];
cx q[30],q[38];
rx(pi*0.005) q[30];
ry(pi*0.5) q[38];
cx q[38],q[30];
sxdg q[38];
s q[38];
cx q[30],q[38];
u3(pi*0.5,pi*1.495,pi*1.0) q[30];
u3(pi*0.5,pi*0.495,pi*1.0) q[38];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[31];
rz(pi*0.495) q[39];
u3(pi*0.5,pi*1.0,pi*1.0) q[31];
u3(pi*0.5,0,pi*1.0) q[39];
sx q[31];
cx q[31],q[39];
rx(pi*0.005) q[31];
ry(pi*0.5) q[39];
cx q[39],q[31];
sxdg q[39];
s q[39];
cx q[31],q[39];
u3(pi*0.5,pi*1.505,0) q[31];
u3(pi*0.5,pi*1.505,pi*1.0) q[39];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[40];
rz(pi*0.495) q[48];
u3(pi*0.5,pi*1.0,pi*1.0) q[40];
u3(pi*0.5,0,pi*1.0) q[48];
sx q[40];
cx q[40],q[48];
rx(pi*0.005) q[40];
ry(pi*0.5) q[48];
cx q[48],q[40];
sxdg q[48];
s q[48];
cx q[40],q[48];
u3(pi*0.5,pi*1.505,0) q[40];
u3(pi*0.5,pi*1.505,pi*1.0) q[48];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[41];
rz(pi*-0.495) q[49];
u3(pi*0.5,0,0) q[41];
u3(pi*0.5,0,pi*1.0) q[49];
sx q[41];
cx q[41],q[49];
rx(pi*0.005) q[41];
ry(pi*0.5) q[49];
cx q[49],q[41];
sxdg q[49];
s q[49];
cx q[41],q[49];
u3(pi*0.5,pi*1.495,pi*1.0) q[41];
u3(pi*0.5,pi*0.495,pi*1.0) q[49];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[42];
rz(pi*-0.495) q[50];
u3(pi*0.5,0,0) q[42];
u3(pi*0.5,0,pi*1.0) q[50];
sx q[42];
cx q[42],q[50];
rx(pi*0.005) q[42];
ry(pi*0.5) q[50];
cx q[50],q[42];
sxdg q[50];
s q[50];
cx q[42],q[50];
u3(pi*0.5,pi*1.495,pi*1.0) q[42];
u3(pi*0.5,pi*0.495,pi*1.0) q[50];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[43];
rz(pi*0.495) q[51];
u3(pi*0.5,pi*1.0,pi*1.0) q[43];
u3(pi*0.5,0,pi*1.0) q[51];
sx q[43];
cx q[43],q[51];
rx(pi*0.005) q[43];
ry(pi*0.5) q[51];
cx q[51],q[43];
sxdg q[51];
s q[51];
cx q[43],q[51];
u3(pi*0.5,pi*1.505,0) q[43];
u3(pi*0.5,pi*1.505,pi*1.0) q[51];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[44];
rz(pi*0.495) q[52];
u3(pi*0.5,pi*1.0,pi*1.0) q[44];
u3(pi*0.5,0,pi*1.0) q[52];
sx q[44];
cx q[44],q[52];
rx(pi*0.005) q[44];
ry(pi*0.5) q[52];
cx q[52],q[44];
sxdg q[52];
s q[52];
cx q[44],q[52];
u3(pi*0.5,pi*1.505,0) q[44];
u3(pi*0.5,pi*1.505,pi*1.0) q[52];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[45];
rz(pi*0.495) q[53];
u3(pi*0.5,pi*1.0,pi*1.0) q[45];
u3(pi*0.5,0,pi*1.0) q[53];
sx q[45];
cx q[45],q[53];
rx(pi*0.005) q[45];
ry(pi*0.5) q[53];
cx q[53],q[45];
sxdg q[53];
s q[53];
cx q[45],q[53];
u3(pi*0.5,pi*1.505,0) q[45];
u3(pi*0.5,pi*1.505,pi*1.0) q[53];

// Gate: ZZ**0.49500000000000005
rz(pi*0.495) q[46];
rz(pi*0.495) q[54];
u3(pi*0.5,pi*1.0,pi*1.0) q[46];
u3(pi*0.5,0,pi*1.0) q[54];
sx q[46];
cx q[46],q[54];
rx(pi*0.005) q[46];
ry(pi*0.5) q[54];
cx q[54],q[46];
sxdg q[54];
s q[54];
cx q[46],q[54];
u3(pi*0.5,pi*1.505,0) q[46];
u3(pi*0.5,pi*1.505,pi*1.0) q[54];

// Gate: ZZ**-0.49500000000000005
rz(pi*-0.495) q[47];
rz(pi*-0.495) q[55];
u3(pi*0.5,0,0) q[47];
u3(pi*0.5,0,pi*1.0) q[55];
sx q[47];
cx q[47],q[55];
rx(pi*0.005) q[47];
ry(pi*0.5) q[55];
cx q[55],q[47];
sxdg q[55];
s q[55];
cx q[47],q[55];
u3(pi*0.5,pi*1.495,pi*1.0) q[47];
u3(pi*0.5,pi*0.495,pi*1.0) q[55];

rx(pi*1.01) q[0];
rx(pi*1.01) q[1];
rx(pi*1.01) q[2];
rx(pi*1.01) q[3];
rx(pi*1.01) q[4];
rx(pi*1.01) q[5];
rx(pi*1.01) q[6];
rx(pi*1.01) q[7];
rx(pi*1.01) q[8];
rx(pi*1.01) q[9];
rx(pi*1.01) q[10];
rx(pi*1.01) q[11];
rx(pi*1.01) q[12];
rx(pi*1.01) q[13];
rx(pi*1.01) q[14];
rx(pi*1.01) q[15];
rx(pi*1.01) q[16];
rx(pi*1.01) q[17];
rx(pi*1.01) q[18];
rx(pi*1.01) q[19];
rx(pi*1.01) q[20];
rx(pi*1.01) q[21];
rx(pi*1.01) q[22];
rx(pi*1.01) q[23];
rx(pi*1.01) q[24];
rx(pi*1.01) q[25];
rx(pi*1.01) q[26];
rx(pi*1.01) q[27];
rx(pi*1.01) q[28];
rx(pi*1.01) q[29];
rx(pi*1.01) q[30];
rx(pi*1.01) q[31];
rx(pi*1.01) q[32];
rx(pi*1.01) q[33];
rx(pi*1.01) q[34];
rx(pi*1.01) q[35];
rx(pi*1.01) q[36];
rx(pi*1.01) q[37];
rx(pi*1.01) q[38];
rx(pi*1.01) q[39];
rx(pi*1.01) q[40];
rx(pi*1.01) q[41];
rx(pi*1.01) q[42];
rx(pi*1.01) q[43];
rx(pi*1.01) q[44];
rx(pi*1.01) q[45];
rx(pi*1.01) q[46];
rx(pi*1.01) q[47];
rx(pi*1.01) q[48];
rx(pi*1.01) q[49];
rx(pi*1.01) q[50];
rx(pi*1.01) q[51];
rx(pi*1.01) q[52];
rx(pi*1.01) q[53];
rx(pi*1.01) q[54];
rx(pi*1.01) q[55];
rx(pi*1.01) q[56];
rx(pi*1.01) q[57];
rx(pi*1.01) q[58];
rx(pi*1.01) q[59];
rx(pi*1.01) q[60];
rx(pi*1.01) q[61];
rx(pi*1.01) q[62];
rx(pi*1.01) q[63];
