OPENQASM 2.0;
include "qelib1.inc";
qreg qr[30];
creg cr[30];
h qr[0];
h qr[1];
h qr[2];
h qr[3];
h qr[4];
h qr[5];
h qr[6];
h qr[7];
h qr[8];
h qr[9];
h qr[10];
h qr[11];
h qr[12];
h qr[13];
h qr[14];
h qr[15];
h qr[16];
h qr[17];
h qr[18];
h qr[19];
h qr[20];
h qr[21];
h qr[22];
h qr[23];
h qr[24];
h qr[25];
h qr[26];
h qr[27];
h qr[28];
h qr[29];
h qr[22];
h qr[26];
cx qr[17], qr[19];
cx qr[23], qr[3];
h qr[27];
h qr[20];
h qr[17];
h qr[16];
cx qr[22], qr[3];
cx qr[21], qr[4];
cx qr[4], qr[22];
cx qr[6], qr[7];
h qr[29];
h qr[24];
cx qr[11], qr[26];
h qr[21];
cx qr[3], qr[0];
cx qr[27], qr[0];
cx qr[13], qr[18];
cx qr[2], qr[14];
cx qr[24], qr[21];
cx qr[21], qr[15];
cx qr[29], qr[22];
h qr[5];
cx qr[14], qr[0];
h qr[6];
h qr[28];
cx qr[27], qr[28];
h qr[9];
cx qr[7], qr[2];