OPENQASM 2.0;
include "qelib1.inc";
qreg q[45];
s q[0];
h q[0];
s q[0];
s q[2];
h q[2];
s q[2];
s q[3];
h q[3];
s q[6];
s q[9];
s q[11];
h q[12];
s q[14];
s q[17];
h q[19];
s q[22];
h q[23];
s q[25];
h q[25];
s q[26];
h q[26];
s q[28];
h q[29];
h q[30];
s q[31];
h q[31];
h q[32];
h q[33];
s q[34];
s q[35];
h q[35];
s q[36];
h q[36];
s q[36];
h q[37];
s q[38];
h q[38];
s q[39];
s q[40];
h q[40];
s q[40];
h q[42];
s q[42];
s q[43];
swap q[20],q[0];
cx q[20],q[6];
cx q[20],q[10];
cx q[20],q[14];
cx q[20],q[17];
cx q[20],q[21];
cx q[20],q[23];
cx q[20],q[28];
cx q[20],q[39];
cx q[20],q[0];
cx q[3],q[20];
cx q[7],q[20];
cx q[16],q[20];
cx q[19],q[20];
cx q[26],q[20];
cx q[27],q[20];
cx q[29],q[20];
cx q[31],q[20];
cx q[32],q[20];
cx q[35],q[20];
cx q[38],q[20];
cx q[44],q[20];
cx q[5],q[30];
cx q[5],q[33];
cx q[5],q[43];
cx q[20],q[5];
h q[5];
cx q[5],q[20];
cx q[8],q[2];
cx q[2],q[20];
cx q[20],q[8];
cx q[11],q[9];
cx q[9],q[20];
cx q[20],q[11];
cx q[18],q[12];
cx q[12],q[20];
cx q[20],q[18];
cx q[24],q[22];
cx q[22],q[20];
cx q[20],q[24];
cx q[34],q[25];
cx q[25],q[20];
cx q[20],q[34];
cx q[37],q[36];
cx q[36],q[20];
cx q[20],q[37];
cx q[42],q[40];
cx q[40],q[20];
cx q[20],q[42];
s q[1];
h q[2];
s q[3];
h q[3];
h q[4];
h q[5];
h q[6];
h q[8];
h q[9];
s q[9];
h q[10];
h q[11];
h q[13];
s q[14];
s q[15];
h q[16];
h q[17];
s q[19];
h q[27];
s q[28];
h q[28];
s q[29];
h q[30];
h q[32];
s q[32];
h q[34];
h q[35];
h q[37];
s q[39];
s q[40];
h q[40];
s q[41];
h q[42];
s q[43];
h q[43];
s q[44];
swap q[21],q[1];
cx q[21],q[0];
cx q[21],q[2];
cx q[21],q[4];
cx q[21],q[8];
cx q[21],q[11];
cx q[21],q[13];
cx q[21],q[14];
cx q[21],q[19];
cx q[21],q[31];
cx q[21],q[34];
cx q[21],q[35];
cx q[21],q[36];
cx q[21],q[39];
cx q[3],q[21];
cx q[10],q[21];
cx q[12],q[21];
cx q[28],q[21];
cx q[30],q[21];
cx q[43],q[21];
cx q[1],q[21];
cx q[17],q[18];
cx q[17],q[37];
cx q[17],q[38];
cx q[17],q[41];
cx q[17],q[42];
cx q[17],q[44];
cx q[21],q[17];
h q[17];
cx q[17],q[21];
cx q[6],q[5];
cx q[5],q[21];
cx q[21],q[6];
cx q[15],q[9];
cx q[9],q[21];
cx q[21],q[15];
cx q[24],q[16];
cx q[16],q[21];
cx q[21],q[24];
cx q[29],q[27];
cx q[27],q[21];
cx q[21],q[29];
cx q[40],q[32];
cx q[32],q[21];
cx q[21],q[40];
h q[0];
s q[0];
s q[1];
s q[3];
h q[3];
s q[3];
s q[4];
h q[4];
s q[4];
h q[5];
h q[9];
s q[10];
s q[11];
h q[11];
s q[12];
s q[13];
h q[13];
s q[14];
s q[16];
h q[19];
s q[25];
s q[28];
h q[28];
h q[30];
s q[31];
s q[32];
h q[32];
s q[32];
h q[33];
s q[34];
h q[35];
h q[37];
s q[37];
h q[38];
s q[40];
h q[40];
s q[42];
h q[42];
s q[43];
h q[44];
cx q[4],q[1];
cx q[4],q[9];
cx q[4],q[16];
cx q[4],q[33];
cx q[4],q[34];
cx q[5],q[4];
cx q[6],q[4];
cx q[8],q[4];
cx q[13],q[4];
cx q[17],q[4];
cx q[19],q[4];
cx q[26],q[4];
cx q[29],q[4];
cx q[36],q[4];
cx q[41],q[4];
cx q[42],q[4];
cx q[14],q[15];
cx q[14],q[23];
cx q[14],q[30];
cx q[14],q[35];
cx q[14],q[38];
cx q[14],q[39];
cx q[14],q[44];
cx q[4],q[14];
h q[14];
cx q[14],q[4];
cx q[3],q[0];
cx q[0],q[4];
cx q[4],q[3];
cx q[11],q[10];
cx q[10],q[4];
cx q[4],q[11];
cx q[18],q[12];
cx q[12],q[4];
cx q[4],q[18];
cx q[25],q[22];
cx q[22],q[4];
cx q[4],q[25];
cx q[31],q[28];
cx q[28],q[4];
cx q[4],q[31];
cx q[37],q[32];
cx q[32],q[4];
cx q[4],q[37];
cx q[43],q[40];
cx q[40],q[4];
cx q[4],q[43];
s q[0];
s q[1];
h q[2];
s q[2];
s q[3];
s q[5];
s q[9];
h q[10];
s q[12];
h q[12];
s q[12];
h q[16];
s q[18];
h q[18];
s q[23];
h q[23];
h q[24];
s q[24];
h q[25];
s q[25];
h q[28];
s q[29];
h q[29];
s q[33];
h q[33];
s q[33];
h q[36];
h q[38];
s q[41];
h q[41];
s q[41];
s q[43];
h q[44];
swap q[26],q[0];
cx q[26],q[3];
cx q[26],q[6];
cx q[26],q[15];
cx q[26],q[16];
cx q[26],q[19];
cx q[26],q[38];
cx q[26],q[43];
cx q[23],q[26];
cx q[28],q[26];
cx q[32],q[26];
cx q[40],q[26];
cx q[42],q[26];
cx q[7],q[13];
cx q[7],q[17];
cx q[7],q[22];
cx q[7],q[36];
cx q[7],q[39];
cx q[7],q[44];
cx q[26],q[7];
h q[7];
cx q[7],q[26];
cx q[2],q[1];
cx q[1],q[26];
cx q[26],q[2];
cx q[9],q[5];
cx q[5],q[26];
cx q[26],q[9];
cx q[12],q[10];
cx q[10],q[26];
cx q[26],q[12];
cx q[24],q[18];
cx q[18],q[26];
cx q[26],q[24];
cx q[27],q[25];
cx q[25],q[26];
cx q[26],q[27];
cx q[33],q[29];
cx q[29],q[26];
cx q[26],q[33];
cx q[35],q[34];
cx q[34],q[26];
cx q[26],q[35];
cx q[41],q[37];
cx q[37],q[26];
cx q[26],q[41];
s q[0];
h q[1];
s q[8];
h q[9];
h q[10];
s q[11];
h q[12];
h q[13];
s q[14];
h q[16];
s q[17];
h q[19];
s q[22];
s q[23];
s q[25];
s q[27];
s q[28];
h q[28];
s q[28];
h q[32];
s q[32];
s q[35];
s q[36];
s q[37];
h q[37];
h q[38];
h q[39];
h q[40];
s q[42];
h q[42];
s q[43];
swap q[35],q[0];
cx q[35],q[1];
cx q[35],q[2];
cx q[35],q[5];
cx q[35],q[11];
cx q[35],q[19];
cx q[35],q[25];
cx q[35],q[30];
cx q[35],q[38];
cx q[35],q[40];
cx q[35],q[43];
cx q[6],q[35];
cx q[12],q[35];
cx q[33],q[35];
cx q[41],q[35];
cx q[42],q[35];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[18];
cx q[8],q[22];
cx q[8],q[23];
cx q[8],q[24];
cx q[8],q[27];
cx q[8],q[29];
cx q[8],q[34];
cx q[8],q[36];
cx q[8],q[39];
cx q[8],q[44];
cx q[8],q[0];
cx q[35],q[8];
h q[8];
cx q[8],q[35];
cx q[13],q[3];
cx q[3],q[35];
cx q[35],q[13];
cx q[16],q[14];
cx q[14],q[35];
cx q[35],q[16];
cx q[28],q[17];
cx q[17],q[35];
cx q[35],q[28];
cx q[37],q[32];
cx q[32],q[35];
cx q[35],q[37];
h q[0];
s q[2];
s q[5];
h q[5];
h q[6];
s q[9];
s q[11];
h q[11];
h q[13];
s q[14];
h q[14];
s q[16];
h q[16];
s q[16];
h q[17];
s q[17];
h q[18];
h q[19];
s q[22];
s q[23];
h q[23];
h q[24];
s q[25];
h q[29];
s q[30];
h q[30];
h q[32];
s q[36];
s q[37];
h q[38];
h q[40];
h q[42];
h q[43];
s q[44];
h q[44];
s q[44];
swap q[30],q[2];
cx q[30],q[0];
cx q[30],q[12];
cx q[30],q[19];
cx q[30],q[22];
cx q[30],q[29];
cx q[30],q[37];
cx q[30],q[38];
cx q[30],q[40];
cx q[6],q[30];
cx q[7],q[30];
cx q[10],q[30];
cx q[11],q[30];
cx q[13],q[30];
cx q[14],q[30];
cx q[27],q[30];
cx q[31],q[30];
cx q[32],q[30];
cx q[33],q[30];
cx q[43],q[30];
cx q[2],q[30];
cx q[9],q[18];
cx q[9],q[25];
cx q[9],q[28];
cx q[9],q[36];
cx q[9],q[41];
cx q[9],q[42];
cx q[30],q[9];
h q[9];
cx q[9],q[30];
cx q[16],q[5];
cx q[5],q[30];
cx q[30],q[16];
cx q[23],q[17];
cx q[17],q[30];
cx q[30],q[23];
cx q[44],q[24];
cx q[24],q[30];
cx q[30],q[44];
h q[1];
s q[2];
h q[2];
s q[2];
s q[6];
s q[10];
h q[12];
s q[14];
h q[15];
h q[17];
h q[18];
s q[19];
h q[19];
s q[22];
h q[22];
h q[27];
s q[28];
s q[29];
h q[29];
s q[29];
h q[31];
s q[32];
h q[32];
h q[33];
s q[36];
s q[37];
h q[37];
s q[38];
h q[39];
h q[40];
s q[41];
h q[41];
s q[41];
s q[44];
h q[44];
swap q[13],q[0];
cx q[13],q[5];
cx q[13],q[9];
cx q[13],q[15];
cx q[13],q[28];
cx q[13],q[40];
cx q[13],q[42];
cx q[1],q[13];
cx q[3],q[13];
cx q[7],q[13];
cx q[12],q[13];
cx q[16],q[13];
cx q[17],q[13];
cx q[19],q[13];
cx q[22],q[13];
cx q[27],q[13];
cx q[31],q[13];
cx q[32],q[13];
cx q[37],q[13];
cx q[39],q[13];
cx q[44],q[13];
cx q[0],q[13];
cx q[6],q[14];
cx q[6],q[23];
cx q[6],q[25];
cx q[6],q[33];
cx q[6],q[36];
cx q[6],q[38];
cx q[13],q[6];
h q[6];
cx q[6],q[13];
cx q[10],q[2];
cx q[2],q[13];
cx q[13],q[10];
cx q[24],q[18];
cx q[18],q[13];
cx q[13],q[24];
cx q[41],q[29];
cx q[29],q[13];
cx q[13],q[41];
s q[0];
h q[0];
s q[0];
s q[1];
h q[1];
s q[1];
s q[3];
h q[5];
s q[5];
s q[8];
h q[8];
h q[9];
s q[11];
h q[17];
s q[18];
h q[18];
s q[19];
s q[22];
h q[22];
h q[24];
s q[29];
h q[29];
h q[31];
h q[32];
s q[36];
s q[38];
h q[39];
s q[41];
h q[44];
swap q[29],q[0];
cx q[29],q[9];
cx q[29],q[34];
cx q[29],q[37];
cx q[29],q[38];
cx q[29],q[44];
cx q[7],q[29];
cx q[18],q[29];
cx q[22],q[29];
cx q[23],q[29];
cx q[25],q[29];
cx q[31],q[29];
cx q[40],q[29];
cx q[42],q[29];
cx q[43],q[29];
cx q[0],q[29];
cx q[6],q[12];
cx q[6],q[16];
cx q[6],q[17];
cx q[6],q[19];
cx q[6],q[24];
cx q[6],q[28];
cx q[6],q[32];
cx q[6],q[33];
cx q[6],q[39];
cx q[6],q[41];
cx q[29],q[6];
h q[6];
cx q[6],q[29];
cx q[2],q[1];
cx q[1],q[29];
cx q[29],q[2];
cx q[5],q[3];
cx q[3],q[29];
cx q[29],q[5];
cx q[11],q[8];
cx q[8],q[29];
cx q[29],q[11];
cx q[36],q[15];
cx q[15],q[29];
cx q[29],q[36];
s q[0];
h q[0];
s q[0];
s q[2];
h q[2];
s q[2];
s q[3];
h q[3];
s q[7];
h q[7];
h q[8];
s q[8];
s q[9];
h q[9];
h q[10];
h q[11];
s q[11];
s q[12];
h q[14];
h q[15];
s q[16];
h q[16];
h q[19];
h q[22];
h q[23];
s q[24];
s q[25];
h q[27];
s q[28];
s q[31];
h q[31];
s q[31];
s q[36];
h q[36];
h q[39];
h q[40];
h q[41];
s q[42];
h q[43];
s q[43];
s q[44];
h q[44];
s q[44];
cx q[0],q[6];
cx q[0],q[15];
cx q[0],q[24];
cx q[0],q[28];
cx q[0],q[39];
cx q[0],q[42];
cx q[7],q[0];
cx q[9],q[0];
cx q[14],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[40],q[0];
cx q[5],q[10];
cx q[5],q[17];
cx q[5],q[22];
cx q[5],q[23];
cx q[5],q[25];
cx q[5],q[34];
cx q[0],q[5];
h q[5];
cx q[5],q[0];
cx q[3],q[2];
cx q[2],q[0];
cx q[0],q[3];
cx q[11],q[8];
cx q[8],q[0];
cx q[0],q[11];
cx q[27],q[12];
cx q[12],q[0];
cx q[0],q[27];
cx q[32],q[31];
cx q[31],q[0];
cx q[0],q[32];
cx q[41],q[36];
cx q[36],q[0];
cx q[0],q[41];
cx q[44],q[43];
cx q[43],q[0];
cx q[0],q[44];
s q[1];
s q[2];
h q[2];
h q[7];
s q[10];
s q[11];
h q[14];
h q[15];
s q[16];
h q[17];
s q[18];
s q[19];
h q[22];
s q[23];
h q[23];
h q[27];
s q[28];
s q[33];
h q[33];
h q[36];
h q[38];
s q[40];
h q[40];
h q[41];
s q[41];
h q[43];
swap q[37],q[1];
cx q[37],q[5];
cx q[37],q[7];
cx q[37],q[11];
cx q[37],q[16];
cx q[37],q[18];
cx q[37],q[19];
cx q[37],q[22];
cx q[37],q[25];
cx q[37],q[34];
cx q[37],q[43];
cx q[17],q[37];
cx q[23],q[37];
cx q[24],q[37];
cx q[27],q[37];
cx q[38],q[37];
cx q[40],q[37];
cx q[1],q[37];
cx q[6],q[10];
cx q[6],q[12];
cx q[6],q[15];
cx q[6],q[36];
cx q[37],q[6];
h q[6];
cx q[6],q[37];
cx q[9],q[2];
cx q[2],q[37];
cx q[37],q[9];
cx q[28],q[14];
cx q[14],q[37];
cx q[37],q[28];
cx q[39],q[33];
cx q[33],q[37];
cx q[37],q[39];
cx q[44],q[41];
cx q[41],q[37];
cx q[37],q[44];
h q[1];
s q[2];
h q[5];
s q[8];
h q[8];
s q[10];
h q[10];
s q[14];
h q[14];
s q[14];
s q[15];
h q[16];
h q[18];
h q[19];
s q[24];
h q[25];
h q[28];
h q[31];
s q[32];
h q[33];
s q[36];
h q[36];
s q[36];
h q[38];
s q[38];
s q[40];
h q[40];
s q[41];
s q[42];
h q[42];
s q[42];
h q[43];
s q[44];
swap q[34],q[1];
cx q[34],q[3];
cx q[34],q[5];
cx q[34],q[17];
cx q[34],q[18];
cx q[34],q[23];
cx q[34],q[24];
cx q[34],q[25];
cx q[34],q[27];
cx q[34],q[31];
cx q[8],q[34];
cx q[16],q[34];
cx q[19],q[34];
cx q[22],q[34];
cx q[33],q[34];
cx q[40],q[34];
cx q[2],q[11];
cx q[2],q[32];
cx q[2],q[41];
cx q[2],q[43];
cx q[2],q[44];
cx q[2],q[1];
cx q[34],q[2];
h q[2];
cx q[2],q[34];
cx q[12],q[10];
cx q[10],q[34];
cx q[34],q[12];
cx q[15],q[14];
cx q[14],q[34];
cx q[34],q[15];
cx q[36],q[28];
cx q[28],q[34];
cx q[34],q[36];
cx q[42],q[38];
cx q[38],q[34];
cx q[34],q[42];
s q[1];
h q[3];
s q[5];
h q[5];
s q[6];
s q[8];
h q[8];
s q[11];
h q[11];
h q[16];
h q[17];
h q[18];
s q[27];
s q[28];
h q[28];
h q[31];
s q[32];
s q[36];
h q[40];
h q[41];
s q[43];
h q[44];
swap q[27],q[3];
cx q[27],q[10];
cx q[27],q[39];
cx q[27],q[43];
cx q[27],q[3];
cx q[5],q[27];
cx q[8],q[27];
cx q[12],q[27];
cx q[14],q[27];
cx q[18],q[27];
cx q[19],q[27];
cx q[23],q[27];
cx q[25],q[27];
cx q[31],q[27];
cx q[38],q[27];
cx q[41],q[27];
cx q[1],q[6];
cx q[1],q[15];
cx q[1],q[16];
cx q[1],q[17];
cx q[1],q[33];
cx q[1],q[36];
cx q[1],q[44];
cx q[27],q[1];
h q[1];
cx q[1],q[27];
cx q[11],q[9];
cx q[9],q[27];
cx q[27],q[11];
cx q[28],q[24];
cx q[24],q[27];
cx q[27],q[28];
cx q[40],q[32];
cx q[32],q[27];
cx q[27],q[40];
s q[5];
h q[5];
h q[6];
h q[8];
h q[9];
s q[9];
h q[11];
h q[14];
h q[15];
s q[17];
h q[23];
s q[24];
h q[24];
h q[31];
s q[32];
h q[33];
s q[36];
h q[36];
h q[39];
h q[40];
s q[41];
s q[42];
h q[42];
s q[42];
s q[43];
h q[44];
swap q[23],q[9];
cx q[23],q[10];
cx q[23],q[14];
cx q[23],q[39];
cx q[2],q[23];
cx q[5],q[23];
cx q[7],q[23];
cx q[8],q[23];
cx q[11],q[23];
cx q[22],q[23];
cx q[24],q[23];
cx q[36],q[23];
cx q[9],q[23];
cx q[6],q[15];
cx q[6],q[17];
cx q[6],q[19];
cx q[6],q[31];
cx q[6],q[32];
cx q[6],q[40];
cx q[6],q[44];
cx q[23],q[6];
h q[6];
cx q[6],q[23];
cx q[33],q[25];
cx q[25],q[23];
cx q[23],q[33];
cx q[41],q[38];
cx q[38],q[23];
cx q[23],q[41];
cx q[43],q[42];
cx q[42],q[23];
cx q[23],q[43];
h q[1];
s q[1];
s q[2];
h q[3];
s q[3];
h q[9];
s q[11];
s q[12];
h q[14];
h q[15];
s q[17];
h q[17];
h q[18];
s q[18];
s q[22];
s q[25];
h q[25];
h q[31];
s q[31];
s q[32];
h q[33];
s q[38];
h q[38];
s q[38];
h q[41];
s q[42];
s q[43];
swap q[17],q[1];
cx q[17],q[11];
cx q[17],q[28];
cx q[17],q[32];
cx q[17],q[33];
cx q[17],q[39];
cx q[17],q[40];
cx q[17],q[42];
cx q[17],q[44];
cx q[5],q[17];
cx q[7],q[17];
cx q[9],q[17];
cx q[15],q[17];
cx q[25],q[17];
cx q[1],q[17];
cx q[2],q[8];
cx q[2],q[10];
cx q[2],q[14];
cx q[2],q[41];
cx q[17],q[2];
h q[2];
cx q[2],q[17];
cx q[12],q[3];
cx q[3],q[17];
cx q[17],q[12];
cx q[18],q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[31],q[22];
cx q[22],q[17];
cx q[17],q[31];
cx q[43],q[38];
cx q[38],q[17];
cx q[17],q[43];
h q[6];
s q[6];
s q[7];
h q[7];
s q[7];
h q[9];
s q[9];
s q[10];
h q[10];
s q[11];
h q[11];
s q[11];
h q[12];
s q[14];
h q[14];
s q[18];
s q[28];
h q[32];
s q[32];
h q[33];
h q[36];
s q[38];
h q[38];
h q[39];
s q[41];
s q[43];
h q[43];
s q[43];
cx q[7],q[2];
cx q[7],q[8];
cx q[7],q[18];
cx q[7],q[28];
cx q[7],q[31];
cx q[7],q[39];
cx q[7],q[41];
cx q[12],q[7];
cx q[15],q[7];
cx q[19],q[7];
cx q[24],q[7];
cx q[36],q[7];
cx q[38],q[7];
cx q[16],q[22];
cx q[16],q[40];
cx q[16],q[44];
cx q[7],q[16];
h q[16];
cx q[16],q[7];
cx q[6],q[1];
cx q[1],q[7];
cx q[7],q[6];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[10];
cx q[14],q[11];
cx q[11],q[7];
cx q[7],q[14];
cx q[33],q[32];
cx q[32],q[7];
cx q[7],q[33];
cx q[43],q[42];
cx q[42],q[7];
cx q[7],q[43];
s q[1];
s q[2];
h q[2];
s q[2];
h q[3];
h q[6];
s q[8];
s q[9];
h q[11];
s q[12];
h q[12];
h q[15];
s q[15];
h q[18];
h q[22];
h q[24];
h q[28];
h q[31];
h q[40];
s q[41];
h q[41];
h q[43];
cx q[5],q[3];
cx q[5],q[9];
cx q[5],q[31];
cx q[5],q[32];
cx q[5],q[33];
cx q[5],q[38];
cx q[5],q[39];
cx q[5],q[40];
cx q[5],q[44];
cx q[6],q[5];
cx q[12],q[5];
cx q[22],q[5];
cx q[11],q[14];
cx q[11],q[16];
cx q[11],q[28];
cx q[11],q[42];
cx q[11],q[43];
cx q[5],q[11];
h q[11];
cx q[11],q[5];
cx q[2],q[1];
cx q[1],q[5];
cx q[5],q[2];
cx q[15],q[8];
cx q[8],q[5];
cx q[5],q[15];
cx q[19],q[18];
cx q[18],q[5];
cx q[5],q[19];
cx q[41],q[24];
cx q[24],q[5];
cx q[5],q[41];
s q[1];
s q[3];
s q[6];
h q[12];
h q[15];
s q[16];
s q[18];
h q[19];
s q[22];
s q[24];
h q[24];
h q[25];
s q[39];
s q[40];
h q[44];
s q[44];
swap q[36],q[1];
cx q[36],q[8];
cx q[36],q[11];
cx q[36],q[16];
cx q[36],q[19];
cx q[36],q[25];
cx q[36],q[28];
cx q[36],q[40];
cx q[15],q[36];
cx q[24],q[36];
cx q[33],q[36];
cx q[1],q[36];
cx q[2],q[38];
cx q[2],q[39];
cx q[36],q[2];
h q[2];
cx q[2],q[36];
cx q[6],q[3];
cx q[3],q[36];
cx q[36],q[6];
cx q[14],q[12];
cx q[12],q[36];
cx q[36],q[14];
cx q[22],q[18];
cx q[18],q[36];
cx q[36],q[22];
cx q[44],q[32];
cx q[32],q[36];
cx q[36],q[44];
s q[1];
h q[1];
s q[1];
h q[2];
s q[3];
h q[3];
h q[8];
s q[10];
h q[10];
s q[12];
h q[14];
s q[14];
s q[16];
h q[16];
h q[18];
s q[19];
h q[19];
s q[19];
s q[24];
h q[24];
s q[24];
s q[31];
s q[33];
h q[33];
s q[33];
h q[42];
s q[42];
h q[43];
swap q[10],q[1];
cx q[10],q[18];
cx q[10],q[22];
cx q[10],q[43];
cx q[15],q[10];
cx q[16],q[10];
cx q[25],q[10];
cx q[28],q[10];
cx q[1],q[10];
cx q[2],q[6];
cx q[2],q[9];
cx q[2],q[11];
cx q[2],q[32];
cx q[10],q[2];
h q[2];
cx q[2],q[10];
cx q[8],q[3];
cx q[3],q[10];
cx q[10],q[8];
cx q[14],q[12];
cx q[12],q[10];
cx q[10],q[14];
cx q[24],q[19];
cx q[19],q[10];
cx q[10],q[24];
cx q[33],q[31];
cx q[31],q[10];
cx q[10],q[33];
cx q[42],q[41];
cx q[41],q[10];
cx q[10],q[42];
h q[1];
h q[2];
s q[6];
h q[9];
h q[11];
h q[12];
s q[15];
h q[15];
s q[15];
s q[18];
h q[18];
s q[18];
s q[19];
s q[25];
h q[25];
h q[31];
h q[38];
s q[39];
s q[42];
h q[42];
h q[43];
h q[44];
s q[44];
swap q[24],q[2];
cx q[24],q[6];
cx q[24],q[43];
cx q[24],q[2];
cx q[1],q[24];
cx q[11],q[24];
cx q[25],q[24];
cx q[31],q[24];
cx q[42],q[24];
cx q[9],q[12];
cx q[9],q[28];
cx q[9],q[33];
cx q[9],q[39];
cx q[24],q[9];
h q[9];
cx q[9],q[24];
cx q[18],q[15];
cx q[15],q[24];
cx q[24],q[18];
cx q[32],q[19];
cx q[19],q[24];
cx q[24],q[32];
cx q[44],q[38];
cx q[38],q[24];
cx q[24],q[44];
h q[2];
h q[3];
h q[6];
s q[8];
h q[8];
s q[8];
s q[9];
h q[9];
s q[9];
h q[11];
s q[12];
h q[12];
h q[15];
h q[18];
h q[25];
h q[28];
s q[38];
h q[38];
s q[38];
s q[40];
h q[42];
swap q[32],q[1];
cx q[32],q[18];
cx q[32],q[39];
cx q[32],q[42];
cx q[32],q[43];
cx q[32],q[1];
cx q[2],q[32];
cx q[12],q[32];
cx q[15],q[32];
cx q[33],q[32];
cx q[3],q[25];
cx q[3],q[28];
cx q[32],q[3];
h q[3];
cx q[3],q[32];
cx q[8],q[6];
cx q[6],q[32];
cx q[32],q[8];
cx q[11],q[9];
cx q[9],q[32];
cx q[32],q[11];
cx q[38],q[31];
cx q[31],q[32];
cx q[32],q[38];
cx q[41],q[40];
cx q[40],q[32];
cx q[32],q[41];
s q[2];
h q[2];
s q[2];
h q[6];
s q[6];
s q[12];
h q[14];
s q[16];
h q[16];
s q[16];
s q[18];
h q[19];
s q[22];
s q[31];
s q[33];
h q[33];
s q[39];
h q[39];
h q[40];
s q[40];
s q[41];
s q[43];
h q[43];
s q[43];
h q[44];
cx q[18],q[9];
cx q[18],q[14];
cx q[18],q[19];
cx q[18],q[25];
cx q[18],q[38];
cx q[15],q[18];
cx q[33],q[18];
cx q[39],q[18];
cx q[42],q[18];
cx q[44],q[18];
cx q[22],q[31];
cx q[22],q[41];
cx q[18],q[22];
h q[22];
cx q[22],q[18];
cx q[2],q[1];
cx q[1],q[18];
cx q[18],q[2];
cx q[11],q[6];
cx q[6],q[18];
cx q[18],q[11];
cx q[16],q[12];
cx q[12],q[18];
cx q[18],q[16];
cx q[43],q[40];
cx q[40],q[18];
cx q[18],q[43];
h q[1];
h q[3];
s q[3];
h q[6];
s q[8];
h q[8];
h q[9];
h q[11];
s q[12];
h q[15];
s q[19];
h q[19];
s q[22];
h q[31];
h q[38];
s q[38];
s q[39];
s q[41];
s q[42];
h q[43];
s q[44];
h q[44];
s q[44];
swap q[2],q[1];
cx q[2],q[14];
cx q[2],q[15];
cx q[2],q[31];
cx q[2],q[40];
cx q[19],q[2];
cx q[1],q[2];
cx q[12],q[41];
cx q[12],q[42];
cx q[12],q[43];
cx q[2],q[12];
h q[12];
cx q[12],q[2];
cx q[6],q[3];
cx q[3],q[2];
cx q[2],q[6];
cx q[9],q[8];
cx q[8],q[2];
cx q[2],q[9];
cx q[22],q[11];
cx q[11],q[2];
cx q[2],q[22];
cx q[38],q[28];
cx q[28],q[2];
cx q[2],q[38];
cx q[44],q[39];
cx q[39],q[2];
cx q[2],q[44];
h q[1];
s q[1];
s q[3];
h q[6];
s q[8];
s q[11];
s q[12];
h q[12];
h q[16];
h q[19];
h q[22];
h q[25];
h q[33];
s q[39];
h q[40];
s q[40];
h q[41];
s q[42];
h q[42];
s q[42];
h q[44];
swap q[6],q[1];
cx q[6],q[8];
cx q[6],q[16];
cx q[6],q[25];
cx q[6],q[39];
cx q[6],q[43];
cx q[12],q[6];
cx q[19],q[6];
cx q[22],q[6];
cx q[33],q[6];
cx q[44],q[6];
cx q[1],q[6];
cx q[3],q[11];
cx q[3],q[14];
cx q[3],q[38];
cx q[6],q[3];
h q[3];
cx q[3],q[6];
cx q[40],q[28];
cx q[28],q[6];
cx q[6],q[40];
cx q[42],q[41];
cx q[41],q[6];
cx q[6],q[42];
h q[1];
h q[3];
s q[8];
h q[11];
s q[11];
s q[14];
h q[16];
s q[22];
h q[22];
s q[25];
h q[25];
s q[31];
h q[40];
s q[41];
h q[41];
h q[43];
s q[43];
h q[44];
cx q[11],q[14];
cx q[11],q[19];
cx q[11],q[42];
cx q[11],q[44];
cx q[22],q[11];
cx q[25],q[11];
cx q[40],q[11];
cx q[41],q[11];
cx q[1],q[8];
cx q[1],q[16];
cx q[11],q[1];
h q[1];
cx q[1],q[11];
cx q[15],q[3];
cx q[3],q[11];
cx q[11],q[15];
cx q[43],q[31];
cx q[31],q[11];
cx q[11],q[43];
h q[1];
s q[3];
s q[8];
h q[8];
h q[12];
h q[14];
s q[14];
h q[16];
s q[33];
h q[33];
s q[38];
h q[38];
s q[38];
s q[39];
h q[39];
s q[42];
h q[44];
swap q[12],q[1];
cx q[12],q[9];
cx q[8],q[12];
cx q[15],q[12];
cx q[16],q[12];
cx q[19],q[12];
cx q[1],q[12];
cx q[22],q[31];
cx q[22],q[40];
cx q[22],q[42];
cx q[12],q[22];
h q[22];
cx q[22],q[12];
cx q[14],q[3];
cx q[3],q[12];
cx q[12],q[14];
cx q[38],q[33];
cx q[33],q[12];
cx q[12],q[38];
cx q[44],q[39];
cx q[39],q[12];
cx q[12],q[44];
s q[1];
h q[1];
h q[3];
s q[9];
s q[14];
h q[14];
s q[14];
h q[15];
s q[16];
h q[19];
h q[22];
h q[25];
s q[28];
h q[28];
s q[28];
h q[33];
s q[39];
h q[39];
s q[39];
h q[40];
cx q[16],q[3];
cx q[16],q[15];
cx q[16],q[25];
cx q[16],q[33];
cx q[16],q[38];
cx q[19],q[16];
cx q[40],q[16];
cx q[41],q[16];
cx q[8],q[9];
cx q[8],q[42];
cx q[16],q[8];
h q[8];
cx q[8],q[16];
cx q[14],q[1];
cx q[1],q[16];
cx q[16],q[14];
cx q[28],q[22];
cx q[22],q[16];
cx q[16],q[28];
cx q[43],q[39];
cx q[39],q[16];
cx q[16],q[43];
h q[1];
s q[3];
h q[3];
s q[9];
s q[15];
h q[15];
s q[15];
s q[22];
h q[22];
s q[22];
s q[25];
s q[31];
h q[33];
s q[33];
h q[38];
s q[39];
h q[39];
s q[42];
h q[42];
s q[42];
h q[43];
s q[44];
h q[44];
swap q[14],q[1];
cx q[14],q[1];
cx q[3],q[14];
cx q[19],q[14];
cx q[38],q[14];
cx q[39],q[14];
cx q[31],q[43];
cx q[14],q[31];
h q[31];
cx q[31],q[14];
cx q[9],q[8];
cx q[8],q[14];
cx q[14],q[9];
cx q[22],q[15];
cx q[15],q[14];
cx q[14],q[22];
cx q[33],q[25];
cx q[25],q[14];
cx q[14],q[33];
cx q[44],q[42];
cx q[42],q[14];
cx q[14],q[44];
h q[1];
h q[8];
h q[9];
h q[15];
s q[28];
h q[31];
s q[40];
s q[42];
cx q[9],q[22];
cx q[9],q[28];
cx q[9],q[31];
cx q[9],q[39];
cx q[3],q[9];
cx q[15],q[9];
cx q[38],q[41];
cx q[38],q[42];
cx q[38],q[43];
cx q[38],q[44];
cx q[9],q[38];
h q[38];
cx q[38],q[9];
cx q[8],q[1];
cx q[1],q[9];
cx q[9],q[8];
cx q[40],q[19];
cx q[19],q[9];
cx q[9],q[40];
s q[1];
s q[25];
s q[28];
h q[28];
s q[28];
h q[31];
h q[38];
s q[38];
s q[39];
h q[39];
s q[40];
h q[40];
h q[41];
s q[42];
h q[42];
swap q[19],q[1];
cx q[19],q[3];
cx q[19],q[8];
cx q[19],q[33];
cx q[19],q[41];
cx q[19],q[1];
cx q[22],q[19];
cx q[39],q[19];
cx q[40],q[19];
cx q[42],q[19];
cx q[43],q[19];
cx q[25],q[31];
cx q[19],q[25];
h q[25];
cx q[25],q[19];
cx q[38],q[28];
cx q[28],q[19];
cx q[19],q[38];
s q[1];
h q[1];
s q[3];
h q[3];
s q[22];
s q[28];
s q[33];
s q[40];
h q[40];
s q[40];
s q[42];
h q[43];
s q[44];
h q[44];
swap q[41],q[1];
cx q[41],q[39];
cx q[41],q[42];
cx q[8],q[41];
cx q[38],q[41];
cx q[25],q[28];
cx q[25],q[33];
cx q[25],q[43];
cx q[41],q[25];
h q[25];
cx q[25],q[41];
cx q[22],q[3];
cx q[3],q[41];
cx q[41],q[22];
cx q[44],q[40];
cx q[40],q[41];
cx q[41],q[44];
s q[1];
h q[1];
s q[1];
h q[3];
s q[15];
h q[15];
s q[15];
s q[22];
h q[22];
h q[31];
h q[38];
h q[39];
h q[40];
h q[42];
s q[42];
h q[43];
s q[44];
swap q[44],q[1];
cx q[44],q[8];
cx q[44],q[40];
cx q[44],q[43];
cx q[44],q[1];
cx q[22],q[44];
cx q[33],q[44];
cx q[3],q[39];
cx q[44],q[3];
h q[3];
cx q[3],q[44];
cx q[31],q[15];
cx q[15],q[44];
cx q[44],q[31];
cx q[42],q[38];
cx q[38],q[44];
cx q[44],q[42];
s q[22];
s q[25];
h q[33];
h q[40];
h q[43];
s q[43];
swap q[39],q[1];
cx q[39],q[22];
cx q[39],q[33];
cx q[39],q[40];
cx q[39],q[1];
cx q[28],q[39];
cx q[31],q[39];
cx q[3],q[8];
cx q[3],q[25];
cx q[39],q[3];
h q[3];
cx q[3],q[39];
cx q[43],q[38];
cx q[38],q[39];
cx q[39],q[43];
h q[3];
s q[22];
s q[25];
h q[31];
h q[38];
s q[40];
s q[42];
h q[42];
s q[42];
h q[43];
swap q[33],q[1];
cx q[33],q[25];
cx q[33],q[43];
cx q[33],q[1];
cx q[8],q[33];
cx q[31],q[33];
cx q[3],q[15];
cx q[3],q[22];
cx q[3],q[40];
cx q[33],q[3];
h q[3];
cx q[3],q[33];
cx q[42],q[38];
cx q[38],q[33];
cx q[33],q[42];
s q[1];
h q[1];
s q[1];
s q[3];
h q[3];
s q[3];
h q[22];
s q[25];
h q[25];
s q[25];
h q[40];
s q[40];
h q[42];
s q[43];
h q[43];
s q[43];
swap q[28],q[1];
cx q[28],q[15];
cx q[28],q[31];
cx q[28],q[38];
cx q[28],q[42];
cx q[8],q[28];
cx q[22],q[28];
cx q[1],q[28];
cx q[25],q[3];
cx q[3],q[28];
cx q[28],q[25];
cx q[43],q[40];
cx q[40],q[28];
cx q[28],q[43];
s q[1];
h q[1];
h q[3];
s q[3];
s q[15];
h q[15];
s q[25];
h q[25];
s q[38];
h q[38];
s q[40];
h q[42];
cx q[1],q[8];
cx q[1],q[22];
cx q[1],q[40];
cx q[1],q[42];
cx q[25],q[1];
cx q[31],q[1];
cx q[38],q[1];
cx q[15],q[3];
cx q[3],q[1];
cx q[1],q[15];
s q[3];
h q[3];
s q[3];
h q[15];
s q[31];
h q[31];
s q[38];
swap q[15],q[3];
cx q[15],q[42];
cx q[31],q[15];
cx q[22],q[25];
cx q[22],q[3];
cx q[15],q[22];
h q[22];
cx q[22],q[15];
cx q[38],q[8];
cx q[8],q[15];
cx q[15],q[38];
h q[3];
h q[8];
h q[25];
h q[38];
h q[43];
swap q[31],q[42];
cx q[31],q[25];
cx q[31],q[38];
cx q[3],q[31];
cx q[22],q[31];
cx q[8],q[43];
cx q[31],q[8];
h q[8];
cx q[8],q[31];
s q[3];
s q[8];
s q[38];
h q[38];
s q[38];
swap q[25],q[3];
cx q[25],q[22];
cx q[3],q[25];
cx q[8],q[42];
cx q[8],q[43];
cx q[25],q[8];
h q[8];
cx q[8],q[25];
cx q[40],q[38];
cx q[38],q[25];
cx q[25],q[40];
s q[3];
h q[8];
s q[22];
h q[22];
s q[22];
h q[38];
s q[40];
h q[40];
s q[40];
s q[42];
s q[43];
swap q[8],q[3];
cx q[8],q[38];
cx q[3],q[8];
cx q[40],q[22];
cx q[22],q[8];
cx q[8],q[40];
cx q[43],q[42];
cx q[42],q[8];
cx q[8],q[43];
s q[3];
h q[3];
s q[38];
s q[40];
s q[42];
h q[42];
s q[43];
cx q[3],q[38];
cx q[3],q[40];
h q[40];
cx q[40],q[3];
cx q[43],q[42];
cx q[42],q[3];
cx q[3],q[43];
h q[40];
h q[42];
h q[43];
swap q[42],q[22];
cx q[43],q[42];
cx q[22],q[42];
cx q[42],q[40];
h q[40];
cx q[40],q[42];
h q[38];
s q[40];
h q[40];
h q[43];
s q[43];
swap q[40],q[22];
cx q[22],q[40];
cx q[43],q[38];
cx q[38],q[40];
cx q[40],q[43];
s q[22];
h q[22];
h q[43];
swap q[38],q[22];
cx q[38],q[43];
cx q[38],q[22];
h q[22];
cx q[22],q[38];
s q[22];
h q[22];
s q[22];
s q[43];
cx q[22],q[43];
h q[43];
cx q[43],q[22];
h q[43];
z q[0];
y q[2];
y q[6];
z q[7];
y q[8];
x q[10];
x q[11];
x q[12];
z q[15];
y q[16];
y q[17];
x q[19];
y q[21];
y q[22];
x q[23];
x q[24];
y q[25];
z q[26];
y q[28];
y q[29];
z q[30];
y q[31];
y q[32];
y q[34];
z q[36];
x q[37];
y q[39];
x q[40];
y q[43];
x q[44];
