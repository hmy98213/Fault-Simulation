OPENQASM 2.0;
include "qelib1.inc";
qreg q[49];
creg c[49];
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
cz q[0],q[1];
cz q[4],q[5];
cz q[9],q[10];
cz q[14],q[15];
cz q[18],q[19];
cz q[23],q[24];
cz q[28],q[29];
cz q[32],q[33];
cz q[37],q[38];
cz q[42],q[43];
cz q[46],q[47];
t q[2];
t q[3];
t q[6];
t q[7];
t q[8];
t q[11];
t q[12];
t q[13];
t q[16];
t q[17];
t q[20];
t q[21];
t q[22];
t q[25];
t q[26];
t q[27];
t q[30];
t q[31];
t q[34];
t q[35];
t q[36];
t q[39];
t q[40];
t q[41];
t q[44];
t q[45];
t q[48];
cz q[7],q[14];
cz q[35],q[42];
cz q[22],q[29];
cz q[9],q[16];
cz q[37],q[44];
cz q[24],q[31];
cz q[11],q[18];
cz q[39],q[46];
cz q[26],q[33];
cz q[13],q[20];
cz q[41],q[48];
rx(pi/2) q[0];
ry(pi/2) q[1];
ry(pi/2) q[4];
rx(pi/2) q[5];
rx(pi/2) q[10];
rx(pi/2) q[15];
ry(pi/2) q[19];
ry(pi/2) q[23];
rx(pi/2) q[28];
rx(pi/2) q[32];
ry(pi/2) q[38];
rx(pi/2) q[43];
ry(pi/2) q[47];
cz q[1],q[2];
cz q[5],q[6];
cz q[10],q[11];
cz q[15],q[16];
cz q[19],q[20];
cz q[24],q[25];
cz q[29],q[30];
cz q[33],q[34];
cz q[38],q[39];
cz q[43],q[44];
cz q[47],q[48];
t q[0];
t q[4];
rx(pi/2) q[7];
rx(pi/2) q[9];
ry(pi/2) q[13];
rx(pi/2) q[14];
ry(pi/2) q[18];
ry(pi/2) q[22];
t q[23];
rx(pi/2) q[26];
t q[28];
rx(pi/2) q[31];
t q[32];
ry(pi/2) q[35];
ry(pi/2) q[37];
rx(pi/2) q[41];
rx(pi/2) q[42];
rx(pi/2) q[46];
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
