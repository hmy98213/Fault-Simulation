OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
x q[24];
z q[23];
x q[21];
z q[21];
x q[20];
z q[18];
x q[16];
x q[15];
x q[14];
x q[13];
z q[12];
x q[8];
z q[8];
x q[7];
x q[6];
z q[6];
z q[5];
x q[4];
z q[2];
x q[1];
z q[0];
h q[24];
sdg q[24];
h q[24];
sdg q[24];
h q[23];
sdg q[23];
h q[23];
sdg q[23];
cx q[24],q[23];
sdg q[23];
cx q[23],q[24];
h q[22];
sdg q[22];
cx q[22],q[23];
h q[22];
cx q[24],q[22];
cx q[23],q[22];
sdg q[22];
cx q[24],q[22];
sdg q[22];
cx q[22],q[24];
cx q[22],q[23];
h q[21];
cx q[21],q[24];
cx q[21],q[23];
h q[21];
h q[21];
h q[20];
sdg q[20];
cx q[20],q[22];
cx q[20],q[21];
h q[20];
cx q[23],q[20];
cx q[22],q[20];
cx q[21],q[20];
swap q[23],q[20];
h q[23];
h q[19];
sdg q[19];
cx q[19],q[23];
cx q[19],q[20];
h q[19];
cx q[23],q[19];
cx q[22],q[19];
cx q[21],q[19];
sdg q[19];
cx q[23],q[19];
cx q[20],q[19];
sdg q[19];
cx q[19],q[24];
cx q[19],q[23];
swap q[21],q[19];
h q[18];
sdg q[18];
cx q[18],q[24];
cx q[18],q[22];
cx q[18],q[20];
h q[18];
cx q[22],q[18];
cx q[20],q[18];
sdg q[18];
cx q[23],q[18];
cx q[22],q[18];
cx q[18],q[22];
cx q[18],q[21];
swap q[20],q[18];
h q[17];
sdg q[17];
cx q[17],q[23];
cx q[17],q[22];
cx q[17],q[21];
cx q[17],q[20];
cx q[17],q[19];
h q[17];
cx q[24],q[17];
cx q[22],q[17];
cx q[20],q[17];
sdg q[17];
cx q[23],q[17];
cx q[21],q[17];
cx q[20],q[17];
cx q[19],q[17];
cx q[17],q[24];
cx q[17],q[19];
swap q[18],q[17];
h q[16];
sdg q[16];
cx q[16],q[24];
cx q[16],q[23];
cx q[16],q[22];
cx q[16],q[21];
cx q[16],q[19];
cx q[16],q[18];
cx q[16],q[17];
h q[16];
cx q[21],q[16];
sdg q[16];
cx q[23],q[16];
cx q[22],q[16];
cx q[21],q[16];
sdg q[16];
cx q[16],q[23];
cx q[16],q[17];
h q[15];
cx q[15],q[24];
cx q[15],q[22];
cx q[15],q[17];
cx q[15],q[16];
h q[15];
cx q[24],q[15];
cx q[23],q[15];
cx q[20],q[15];
cx q[18],q[15];
cx q[17],q[15];
cx q[16],q[15];
sdg q[15];
cx q[24],q[15];
cx q[21],q[15];
cx q[19],q[15];
sdg q[15];
cx q[15],q[24];
cx q[15],q[21];
cx q[15],q[20];
cx q[15],q[18];
cx q[15],q[16];
h q[14];
sdg q[14];
cx q[14],q[24];
cx q[14],q[23];
cx q[14],q[20];
cx q[14],q[18];
h q[14];
cx q[23],q[14];
cx q[16],q[14];
cx q[15],q[14];
sdg q[14];
cx q[19],q[14];
cx q[18],q[14];
cx q[17],q[14];
cx q[16],q[14];
cx q[15],q[14];
sdg q[14];
cx q[14],q[24];
cx q[14],q[23];
cx q[14],q[22];
cx q[14],q[21];
cx q[14],q[20];
cx q[14],q[18];
cx q[14],q[17];
swap q[16],q[14];
h q[13];
sdg q[13];
cx q[13],q[19];
cx q[13],q[17];
cx q[13],q[14];
h q[13];
cx q[22],q[13];
cx q[20],q[13];
cx q[18],q[13];
cx q[17],q[13];
cx q[16],q[13];
sdg q[13];
cx q[21],q[13];
cx q[20],q[13];
cx q[15],q[13];
cx q[13],q[24];
cx q[13],q[23];
cx q[13],q[22];
cx q[13],q[21];
cx q[13],q[18];
cx q[13],q[16];
cx q[13],q[15];
h q[12];
sdg q[12];
cx q[12],q[22];
cx q[12],q[21];
cx q[12],q[19];
h q[12];
cx q[24],q[12];
cx q[22],q[12];
cx q[20],q[12];
cx q[18],q[12];
cx q[17],q[12];
cx q[14],q[12];
cx q[13],q[12];
sdg q[12];
cx q[24],q[12];
cx q[23],q[12];
cx q[22],q[12];
cx q[17],q[12];
cx q[16],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[23];
cx q[12],q[22];
cx q[12],q[21];
cx q[12],q[20];
cx q[12],q[19];
cx q[12],q[18];
swap q[16],q[12];
h q[11];
sdg q[11];
cx q[11],q[24];
cx q[11],q[23];
cx q[11],q[21];
cx q[11],q[18];
cx q[11],q[14];
cx q[11],q[13];
cx q[11],q[12];
h q[11];
cx q[24],q[11];
cx q[23],q[11];
cx q[22],q[11];
cx q[21],q[11];
cx q[18],q[11];
cx q[17],q[11];
cx q[15],q[11];
cx q[14],q[11];
sdg q[11];
cx q[23],q[11];
cx q[22],q[11];
cx q[21],q[11];
cx q[20],q[11];
cx q[19],q[11];
cx q[16],q[11];
cx q[15],q[11];
cx q[14],q[11];
cx q[13],q[11];
cx q[12],q[11];
cx q[11],q[22];
cx q[11],q[21];
cx q[11],q[20];
cx q[11],q[19];
cx q[11],q[18];
cx q[11],q[17];
cx q[11],q[16];
cx q[11],q[15];
h q[10];
cx q[10],q[24];
cx q[10],q[23];
cx q[10],q[22];
cx q[10],q[21];
cx q[10],q[19];
cx q[10],q[18];
cx q[10],q[17];
cx q[10],q[16];
cx q[10],q[14];
cx q[10],q[13];
cx q[10],q[12];
cx q[10],q[11];
h q[10];
cx q[24],q[10];
cx q[22],q[10];
cx q[21],q[10];
cx q[19],q[10];
cx q[18],q[10];
cx q[17],q[10];
cx q[16],q[10];
cx q[15],q[10];
cx q[13],q[10];
cx q[12],q[10];
cx q[11],q[10];
sdg q[10];
cx q[22],q[10];
cx q[19],q[10];
cx q[17],q[10];
cx q[16],q[10];
cx q[15],q[10];
cx q[12],q[10];
cx q[11],q[10];
sdg q[10];
cx q[10],q[23];
cx q[10],q[21];
cx q[10],q[20];
cx q[10],q[18];
cx q[10],q[17];
cx q[10],q[14];
cx q[10],q[13];
cx q[10],q[11];
h q[9];
cx q[9],q[24];
cx q[9],q[22];
cx q[9],q[19];
cx q[9],q[14];
cx q[9],q[10];
h q[9];
cx q[24],q[9];
cx q[22],q[9];
cx q[21],q[9];
cx q[20],q[9];
cx q[19],q[9];
cx q[17],q[9];
cx q[16],q[9];
cx q[14],q[9];
cx q[12],q[9];
cx q[10],q[9];
sdg q[9];
cx q[24],q[9];
cx q[22],q[9];
cx q[21],q[9];
cx q[18],q[9];
cx q[14],q[9];
cx q[13],q[9];
cx q[12],q[9];
cx q[11],q[9];
cx q[10],q[9];
cx q[9],q[24];
cx q[9],q[22];
cx q[9],q[21];
cx q[9],q[19];
cx q[9],q[18];
cx q[9],q[15];
cx q[9],q[13];
swap q[12],q[9];
h q[8];
cx q[8],q[22];
cx q[8],q[16];
cx q[8],q[15];
cx q[8],q[14];
cx q[8],q[12];
cx q[8],q[11];
cx q[8],q[9];
h q[8];
cx q[23],q[8];
cx q[22],q[8];
cx q[20],q[8];
cx q[19],q[8];
cx q[18],q[8];
cx q[17],q[8];
cx q[16],q[8];
cx q[15],q[8];
cx q[11],q[8];
cx q[10],q[8];
cx q[9],q[8];
sdg q[8];
cx q[24],q[8];
cx q[20],q[8];
cx q[17],q[8];
cx q[16],q[8];
cx q[12],q[8];
cx q[10],q[8];
cx q[9],q[8];
sdg q[8];
cx q[8],q[24];
cx q[8],q[22];
cx q[8],q[21];
cx q[8],q[17];
cx q[8],q[16];
cx q[8],q[14];
cx q[8],q[13];
cx q[8],q[12];
cx q[8],q[11];
cx q[8],q[10];
h q[7];
sdg q[7];
cx q[7],q[21];
cx q[7],q[20];
cx q[7],q[19];
cx q[7],q[12];
cx q[7],q[10];
cx q[7],q[9];
h q[7];
cx q[24],q[7];
cx q[23],q[7];
cx q[22],q[7];
cx q[21],q[7];
cx q[16],q[7];
cx q[14],q[7];
cx q[12],q[7];
cx q[11],q[7];
cx q[9],q[7];
sdg q[7];
cx q[23],q[7];
cx q[22],q[7];
cx q[21],q[7];
cx q[20],q[7];
cx q[17],q[7];
cx q[14],q[7];
cx q[11],q[7];
cx q[8],q[7];
cx q[7],q[22];
cx q[7],q[19];
cx q[7],q[18];
cx q[7],q[17];
cx q[7],q[16];
cx q[7],q[15];
swap q[11],q[7];
h q[6];
cx q[6],q[24];
cx q[6],q[21];
cx q[6],q[20];
cx q[6],q[19];
cx q[6],q[18];
cx q[6],q[17];
cx q[6],q[16];
cx q[6],q[14];
cx q[6],q[13];
cx q[6],q[11];
cx q[6],q[10];
cx q[6],q[9];
cx q[6],q[8];
cx q[6],q[7];
h q[6];
cx q[23],q[6];
cx q[22],q[6];
cx q[21],q[6];
cx q[20],q[6];
cx q[13],q[6];
cx q[12],q[6];
cx q[11],q[6];
cx q[10],q[6];
cx q[9],q[6];
cx q[8],q[6];
cx q[7],q[6];
sdg q[6];
cx q[22],q[6];
cx q[20],q[6];
cx q[19],q[6];
cx q[17],q[6];
cx q[14],q[6];
cx q[12],q[6];
cx q[10],q[6];
cx q[9],q[6];
cx q[8],q[6];
cx q[7],q[6];
cx q[6],q[23];
cx q[6],q[21];
cx q[6],q[19];
cx q[6],q[16];
cx q[6],q[15];
cx q[6],q[13];
cx q[6],q[12];
cx q[6],q[11];
cx q[6],q[10];
cx q[6],q[8];
cx q[6],q[7];
h q[5];
cx q[5],q[24];
cx q[5],q[23];
cx q[5],q[22];
cx q[5],q[20];
cx q[5],q[18];
cx q[5],q[15];
cx q[5],q[12];
cx q[5],q[10];
cx q[5],q[9];
h q[5];
cx q[22],q[5];
cx q[21],q[5];
cx q[20],q[5];
cx q[18],q[5];
cx q[17],q[5];
cx q[16],q[5];
cx q[14],q[5];
cx q[12],q[5];
cx q[11],q[5];
cx q[9],q[5];
cx q[8],q[5];
cx q[7],q[5];
sdg q[5];
cx q[24],q[5];
cx q[23],q[5];
cx q[18],q[5];
cx q[17],q[5];
cx q[15],q[5];
cx q[14],q[5];
cx q[13],q[5];
cx q[12],q[5];
cx q[11],q[5];
cx q[10],q[5];
cx q[8],q[5];
cx q[7],q[5];
sdg q[5];
cx q[5],q[21];
cx q[5],q[19];
cx q[5],q[17];
cx q[5],q[14];
cx q[5],q[12];
cx q[5],q[10];
cx q[5],q[8];
swap q[6],q[5];
h q[4];
sdg q[4];
cx q[4],q[24];
cx q[4],q[23];
cx q[4],q[21];
cx q[4],q[19];
cx q[4],q[14];
cx q[4],q[13];
cx q[4],q[10];
cx q[4],q[8];
cx q[4],q[7];
cx q[4],q[6];
cx q[4],q[5];
h q[4];
cx q[24],q[4];
cx q[22],q[4];
cx q[21],q[4];
cx q[19],q[4];
cx q[16],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[11],q[4];
cx q[8],q[4];
cx q[7],q[4];
sdg q[4];
cx q[24],q[4];
cx q[23],q[4];
cx q[22],q[4];
cx q[19],q[4];
cx q[18],q[4];
cx q[17],q[4];
cx q[15],q[4];
cx q[14],q[4];
cx q[9],q[4];
cx q[8],q[4];
cx q[7],q[4];
cx q[6],q[4];
sdg q[4];
cx q[4],q[21];
cx q[4],q[20];
cx q[4],q[15];
cx q[4],q[12];
cx q[4],q[11];
cx q[4],q[10];
cx q[4],q[7];
swap q[5],q[4];
h q[3];
cx q[3],q[21];
cx q[3],q[19];
cx q[3],q[18];
cx q[3],q[17];
cx q[3],q[16];
cx q[3],q[13];
cx q[3],q[12];
cx q[3],q[11];
cx q[3],q[10];
cx q[3],q[7];
cx q[3],q[5];
h q[3];
cx q[22],q[3];
cx q[21],q[3];
cx q[19],q[3];
cx q[15],q[3];
cx q[13],q[3];
cx q[11],q[3];
cx q[10],q[3];
cx q[9],q[3];
cx q[8],q[3];
cx q[7],q[3];
cx q[5],q[3];
cx q[4],q[3];
sdg q[3];
cx q[24],q[3];
cx q[21],q[3];
cx q[20],q[3];
cx q[15],q[3];
cx q[14],q[3];
cx q[13],q[3];
cx q[11],q[3];
cx q[9],q[3];
cx q[8],q[3];
cx q[7],q[3];
cx q[6],q[3];
cx q[3],q[20];
cx q[3],q[19];
cx q[3],q[18];
cx q[3],q[17];
cx q[3],q[14];
cx q[3],q[12];
cx q[3],q[8];
cx q[3],q[6];
cx q[3],q[4];
h q[2];
sdg q[2];
cx q[2],q[24];
cx q[2],q[22];
cx q[2],q[20];
cx q[2],q[19];
cx q[2],q[18];
cx q[2],q[17];
cx q[2],q[16];
cx q[2],q[15];
cx q[2],q[14];
cx q[2],q[12];
cx q[2],q[9];
cx q[2],q[6];
cx q[2],q[3];
h q[2];
cx q[22],q[2];
cx q[21],q[2];
cx q[19],q[2];
cx q[17],q[2];
cx q[14],q[2];
cx q[12],q[2];
cx q[9],q[2];
cx q[4],q[2];
cx q[3],q[2];
sdg q[2];
cx q[24],q[2];
cx q[23],q[2];
cx q[22],q[2];
cx q[20],q[2];
cx q[18],q[2];
cx q[17],q[2];
cx q[16],q[2];
cx q[15],q[2];
cx q[13],q[2];
cx q[12],q[2];
cx q[9],q[2];
cx q[8],q[2];
cx q[6],q[2];
cx q[5],q[2];
cx q[4],q[2];
sdg q[2];
cx q[2],q[24];
cx q[2],q[23];
cx q[2],q[21];
cx q[2],q[19];
cx q[2],q[18];
cx q[2],q[15];
cx q[2],q[10];
cx q[2],q[9];
cx q[2],q[8];
cx q[2],q[7];
cx q[2],q[6];
cx q[2],q[5];
cx q[2],q[4];
swap q[3],q[2];
h q[1];
cx q[1],q[23];
cx q[1],q[22];
cx q[1],q[21];
cx q[1],q[18];
cx q[1],q[17];
cx q[1],q[15];
cx q[1],q[12];
cx q[1],q[11];
cx q[1],q[10];
cx q[1],q[8];
cx q[1],q[5];
cx q[1],q[4];
h q[1];
cx q[24],q[1];
cx q[23],q[1];
cx q[21],q[1];
cx q[20],q[1];
cx q[18],q[1];
cx q[17],q[1];
cx q[12],q[1];
cx q[11],q[1];
cx q[9],q[1];
cx q[6],q[1];
cx q[3],q[1];
sdg q[1];
cx q[24],q[1];
cx q[21],q[1];
cx q[19],q[1];
cx q[16],q[1];
cx q[12],q[1];
cx q[11],q[1];
cx q[10],q[1];
cx q[9],q[1];
cx q[8],q[1];
cx q[6],q[1];
cx q[5],q[1];
cx q[4],q[1];
cx q[3],q[1];
cx q[2],q[1];
sdg q[1];
cx q[1],q[24];
cx q[1],q[18];
cx q[1],q[17];
cx q[1],q[15];
cx q[1],q[14];
cx q[1],q[12];
cx q[1],q[11];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[6];
cx q[1],q[5];
cx q[1],q[4];
cx q[1],q[3];
swap q[2],q[1];
h q[0];
sdg q[0];
cx q[0],q[23];
cx q[0],q[22];
cx q[0],q[21];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[18];
cx q[0],q[16];
cx q[0],q[15];
cx q[0],q[14];
cx q[0],q[13];
cx q[0],q[12];
cx q[0],q[11];
cx q[0],q[8];
cx q[0],q[6];
cx q[0],q[3];
cx q[0],q[2];
cx q[0],q[1];
h q[0];
cx q[22],q[0];
cx q[21],q[0];
cx q[20],q[0];
cx q[19],q[0];
cx q[18],q[0];
cx q[16],q[0];
cx q[15],q[0];
cx q[14],q[0];
cx q[12],q[0];
cx q[11],q[0];
cx q[4],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[1],q[0];
sdg q[0];
cx q[21],q[0];
cx q[20],q[0];
cx q[17],q[0];
cx q[16],q[0];
cx q[14],q[0];
cx q[13],q[0];
cx q[12],q[0];
cx q[11],q[0];
cx q[9],q[0];
cx q[7],q[0];
cx q[3],q[0];
cx q[2],q[0];
sdg q[0];
cx q[0],q[24];
cx q[0],q[23];
cx q[0],q[21];
cx q[0],q[20];
cx q[0],q[19];
cx q[0],q[18];
cx q[0],q[15];
cx q[0],q[14];
cx q[0],q[13];
cx q[0],q[12];
cx q[0],q[11];
cx q[0],q[5];
cx q[0],q[4];
swap q[1],q[0];
