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
h qr[15];
h qr[2];
cx qr[12], qr[9];
h qr[10];
cx qr[28], qr[10];
cx qr[4], qr[0];
cx qr[11], qr[1];
cx qr[20], qr[8];
h qr[16];
h qr[20];
h qr[20];
h qr[28];
cx qr[7], qr[4];
h qr[4];
h qr[0];
cx qr[2], qr[20];
h qr[19];
h qr[2];
cx qr[10], qr[22];
cx qr[27], qr[3];
cx qr[22], qr[23];
h qr[28];
h qr[10];
cx qr[10], qr[12];
h qr[12];
cx qr[8], qr[25];
cx qr[6], qr[19];
cx qr[14], qr[15];
cx qr[0], qr[6];
h qr[6];