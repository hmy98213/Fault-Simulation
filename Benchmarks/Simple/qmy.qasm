OPENQASM 2.0;
include "qelib1.inc";
qreg qr[3];
creg cr[3];
ccx qr[0], qr[1], qr[2];

