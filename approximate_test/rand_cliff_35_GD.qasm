OPENQASM 2.0;
include "qelib1.inc";
qreg q[35];
s q[0];
h q[0];
s q[1];
h q[1];
h q[4];
h q[7];
s q[8];
h q[9];
s q[11];
h q[11];
s q[11];
h q[16];
s q[18];
s q[19];
h q[19];
s q[23];
s q[25];
h q[26];
s q[28];
s q[29];
h q[29];
s q[30];
h q[32];
s q[33];
h q[34];
swap q[9],q[0];
cx q[9],q[7];
cx q[12],q[9];
cx q[15],q[9];
cx q[19],q[9];
cx q[26],q[9];
cx q[29],q[9];
cx q[4],q[18];
cx q[4],q[20];
cx q[4],q[24];
cx q[4],q[27];
cx q[4],q[28];
cx q[4],q[0];
cx q[9],q[4];
h q[4];
cx q[4],q[9];
cx q[2],q[1];
cx q[1],q[9];
cx q[9],q[2];
cx q[8],q[5];
cx q[5],q[9];
cx q[9],q[8];
cx q[16],q[11];
cx q[11],q[9];
cx q[9],q[16];
cx q[22],q[17];
cx q[17],q[9];
cx q[9],q[22];
cx q[25],q[23];
cx q[23],q[9];
cx q[9],q[25];
cx q[32],q[30];
cx q[30],q[9];
cx q[9],q[32];
cx q[34],q[33];
cx q[33],q[9];
cx q[9],q[34];
h q[0];
s q[0];
s q[1];
h q[2];
s q[2];
h q[4];
h q[5];
s q[7];
h q[7];
s q[7];
h q[8];
s q[10];
h q[10];
s q[11];
h q[11];
h q[12];
s q[12];
s q[13];
s q[14];
h q[14];
h q[15];
h q[16];
h q[18];
s q[19];
h q[19];
h q[20];
h q[24];
s q[26];
s q[27];
s q[28];
s q[29];
h q[29];
s q[29];
s q[30];
h q[31];
s q[32];
h q[32];
s q[33];
h q[33];
h q[34];
swap q[5],q[0];
cx q[5],q[8];
cx q[5],q[13];
cx q[5],q[21];
cx q[5],q[31];
cx q[5],q[0];
cx q[3],q[5];
cx q[6],q[5];
cx q[10],q[5];
cx q[11],q[5];
cx q[14],q[5];
cx q[16],q[5];
cx q[18],q[5];
cx q[19],q[5];
cx q[24],q[5];
cx q[32],q[5];
cx q[33],q[5];
cx q[15],q[23];
cx q[15],q[26];
cx q[15],q[34];
cx q[5],q[15];
h q[15];
cx q[15],q[5];
cx q[2],q[1];
cx q[1],q[5];
cx q[5],q[2];
cx q[7],q[4];
cx q[4],q[5];
cx q[5],q[7];
cx q[20],q[12];
cx q[12],q[5];
cx q[5],q[20];
cx q[28],q[27];
cx q[27],q[5];
cx q[5],q[28];
cx q[30],q[29];
cx q[29],q[5];
cx q[5],q[30];
s q[0];
s q[1];
h q[3];
s q[4];
h q[4];
s q[6];
h q[12];
h q[15];
s q[17];
h q[18];
s q[19];
s q[21];
s q[23];
h q[23];
h q[24];
s q[24];
h q[28];
h q[29];
h q[30];
s q[30];
h q[31];
s q[31];
h q[34];
swap q[25],q[17];
cx q[25],q[0];
cx q[25],q[1];
cx q[25],q[6];
cx q[25],q[7];
cx q[25],q[11];
cx q[25],q[12];
cx q[25],q[13];
cx q[25],q[16];
cx q[4],q[25];
cx q[8],q[25];
cx q[14],q[25];
cx q[18],q[25];
cx q[23],q[25];
cx q[28],q[25];
cx q[33],q[25];
cx q[34],q[25];
cx q[2],q[3];
cx q[2],q[15];
cx q[2],q[20];
cx q[2],q[32];
cx q[25],q[2];
h q[2];
cx q[2],q[25];
cx q[21],q[19];
cx q[19],q[25];
cx q[25],q[21];
cx q[29],q[24];
cx q[24],q[25];
cx q[25],q[29];
cx q[31],q[30];
cx q[30],q[25];
cx q[25],q[31];
h q[0];
s q[1];
h q[1];
s q[3];
h q[4];
s q[4];
s q[7];
h q[7];
h q[10];
s q[11];
h q[11];
s q[11];
h q[13];
h q[16];
h q[17];
h q[18];
s q[19];
h q[20];
s q[20];
s q[22];
s q[23];
h q[23];
s q[24];
s q[28];
h q[28];
h q[29];
h q[34];
swap q[22],q[0];
cx q[22],q[2];
cx q[22],q[6];
cx q[22],q[8];
cx q[22],q[18];
cx q[22],q[19];
cx q[22],q[24];
cx q[22],q[26];
cx q[22],q[0];
cx q[1],q[22];
cx q[7],q[22];
cx q[13],q[22];
cx q[15],q[22];
cx q[23],q[22];
cx q[28],q[22];
cx q[29],q[22];
cx q[34],q[22];
cx q[3],q[12];
cx q[3],q[16];
cx q[3],q[21];
cx q[3],q[27];
cx q[3],q[30];
cx q[22],q[3];
h q[3];
cx q[3],q[22];
cx q[10],q[4];
cx q[4],q[22];
cx q[22],q[10];
cx q[14],q[11];
cx q[11],q[22];
cx q[22],q[14];
cx q[20],q[17];
cx q[17],q[22];
cx q[22],q[20];
cx q[33],q[31];
cx q[31],q[22];
cx q[22],q[33];
s q[0];
h q[0];
h q[1];
s q[1];
s q[3];
s q[4];
s q[6];
h q[6];
s q[7];
h q[8];
s q[8];
s q[10];
s q[11];
s q[12];
h q[12];
s q[13];
h q[14];
s q[14];
h q[15];
h q[16];
s q[16];
s q[17];
s q[19];
h q[19];
s q[26];
h q[26];
s q[26];
s q[27];
h q[29];
s q[30];
h q[30];
s q[31];
h q[31];
h q[32];
h q[34];
swap q[23],q[0];
cx q[23],q[3];
cx q[23],q[4];
cx q[23],q[11];
cx q[23],q[13];
cx q[23],q[17];
cx q[23],q[20];
cx q[23],q[24];
cx q[23],q[34];
cx q[6],q[23];
cx q[12],q[23];
cx q[29],q[23];
cx q[7],q[10];
cx q[7],q[15];
cx q[7],q[27];
cx q[7],q[28];
cx q[23],q[7];
h q[7];
cx q[7],q[23];
cx q[8],q[1];
cx q[1],q[23];
cx q[23],q[8];
cx q[16],q[14];
cx q[14],q[23];
cx q[23],q[16];
cx q[19],q[18];
cx q[18],q[23];
cx q[23],q[19];
cx q[30],q[26];
cx q[26],q[23];
cx q[23],q[30];
cx q[32],q[31];
cx q[31],q[23];
cx q[23],q[32];
s q[2];
h q[2];
s q[3];
h q[3];
s q[8];
s q[10];
h q[10];
h q[11];
s q[12];
s q[13];
h q[15];
s q[17];
s q[18];
s q[19];
h q[19];
s q[19];
h q[21];
s q[26];
h q[27];
s q[29];
h q[29];
h q[30];
s q[31];
h q[32];
swap q[0],q[4];
cx q[0],q[8];
cx q[0],q[12];
cx q[0],q[16];
cx q[0],q[18];
cx q[0],q[26];
cx q[0],q[27];
cx q[2],q[0];
cx q[3],q[0];
cx q[10],q[0];
cx q[11],q[0];
cx q[24],q[0];
cx q[29],q[0];
cx q[30],q[0];
cx q[4],q[0];
cx q[1],q[6];
cx q[1],q[7];
cx q[1],q[13];
cx q[1],q[15];
cx q[1],q[17];
cx q[1],q[20];
cx q[1],q[21];
cx q[1],q[28];
cx q[1],q[31];
cx q[0],q[1];
h q[1];
cx q[1],q[0];
cx q[32],q[19];
cx q[19],q[0];
cx q[0],q[32];
s q[1];
s q[4];
h q[6];
h q[11];
s q[13];
h q[13];
s q[13];
s q[14];
h q[14];
s q[14];
s q[15];
h q[15];
s q[19];
h q[21];
h q[24];
s q[30];
h q[30];
s q[31];
h q[31];
s q[31];
s q[32];
h q[32];
s q[32];
h q[34];
swap q[26],q[1];
cx q[26],q[11];
cx q[26],q[12];
cx q[26],q[19];
cx q[26],q[21];
cx q[26],q[28];
cx q[2],q[26];
cx q[6],q[26];
cx q[7],q[26];
cx q[20],q[26];
cx q[27],q[26];
cx q[30],q[26];
cx q[8],q[16];
cx q[8],q[17];
cx q[8],q[24];
cx q[8],q[34];
cx q[8],q[1];
cx q[26],q[8];
h q[8];
cx q[8],q[26];
cx q[13],q[4];
cx q[4],q[26];
cx q[26],q[13];
cx q[15],q[14];
cx q[14],q[26];
cx q[26],q[15];
cx q[32],q[31];
cx q[31],q[26];
cx q[26],q[32];
h q[2];
h q[3];
h q[4];
h q[7];
h q[8];
s q[10];
h q[11];
h q[12];
s q[12];
h q[13];
h q[14];
s q[15];
h q[18];
s q[19];
s q[24];
h q[24];
s q[24];
h q[27];
s q[29];
h q[29];
s q[32];
h q[32];
swap q[21],q[1];
cx q[21],q[2];
cx q[21],q[3];
cx q[21],q[4];
cx q[21],q[8];
cx q[21],q[15];
cx q[21],q[16];
cx q[21],q[17];
cx q[21],q[19];
cx q[21],q[33];
cx q[11],q[21];
cx q[20],q[21];
cx q[28],q[21];
cx q[29],q[21];
cx q[32],q[21];
cx q[34],q[21];
cx q[1],q[21];
cx q[6],q[7];
cx q[6],q[10];
cx q[6],q[13];
cx q[6],q[14];
cx q[6],q[27];
cx q[21],q[6];
h q[6];
cx q[6],q[21];
cx q[18],q[12];
cx q[12],q[21];
cx q[21],q[18];
cx q[30],q[24];
cx q[24],q[21];
cx q[21],q[30];
s q[4];
h q[4];
s q[6];
h q[6];
s q[8];
s q[10];
h q[11];
h q[12];
h q[13];
h q[18];
h q[19];
s q[20];
s q[24];
h q[27];
s q[27];
h q[28];
s q[31];
h q[31];
s q[32];
s q[34];
h q[34];
s q[34];
cx q[6],q[10];
cx q[6],q[12];
cx q[6],q[14];
cx q[6],q[18];
cx q[6],q[20];
cx q[4],q[6];
cx q[13],q[6];
cx q[19],q[6];
cx q[30],q[6];
cx q[31],q[6];
cx q[2],q[11];
cx q[2],q[16];
cx q[2],q[24];
cx q[2],q[28];
cx q[2],q[32];
cx q[6],q[2];
h q[2];
cx q[2],q[6];
cx q[15],q[8];
cx q[8],q[6];
cx q[6],q[15];
cx q[34],q[27];
cx q[27],q[6];
cx q[6],q[34];
s q[1];
h q[1];
h q[3];
s q[7];
h q[7];
s q[8];
h q[8];
s q[11];
h q[11];
s q[11];
h q[12];
s q[13];
s q[15];
s q[16];
h q[16];
h q[17];
h q[19];
h q[20];
h q[24];
s q[24];
h q[27];
s q[28];
h q[28];
s q[30];
h q[33];
s q[33];
s q[34];
h q[34];
s q[34];
swap q[20],q[1];
cx q[20],q[3];
cx q[20],q[14];
cx q[20],q[17];
cx q[20],q[30];
cx q[20],q[1];
cx q[2],q[20];
cx q[8],q[20];
cx q[19],q[20];
cx q[27],q[20];
cx q[12],q[13];
cx q[12],q[18];
cx q[12],q[29];
cx q[20],q[12];
h q[12];
cx q[12],q[20];
cx q[11],q[7];
cx q[7],q[20];
cx q[20],q[11];
cx q[16],q[15];
cx q[15],q[20];
cx q[20],q[16];
cx q[28],q[24];
cx q[24],q[20];
cx q[20],q[28];
cx q[34],q[33];
cx q[33],q[20];
cx q[20],q[34];
s q[1];
h q[1];
h q[3];
s q[4];
s q[7];
h q[7];
s q[8];
h q[8];
s q[10];
h q[10];
h q[11];
h q[12];
h q[15];
s q[15];
s q[16];
h q[16];
h q[18];
s q[19];
s q[24];
h q[24];
s q[28];
h q[28];
s q[28];
h q[29];
s q[30];
h q[30];
h q[31];
h q[34];
swap q[2],q[1];
cx q[2],q[13];
cx q[2],q[19];
cx q[2],q[34];
cx q[2],q[1];
cx q[7],q[2];
cx q[10],q[2];
cx q[12],q[2];
cx q[30],q[2];
cx q[33],q[2];
cx q[3],q[4];
cx q[3],q[11];
cx q[3],q[18];
cx q[3],q[29];
cx q[2],q[3];
h q[3];
cx q[3],q[2];
cx q[15],q[8];
cx q[8],q[2];
cx q[2],q[15];
cx q[24],q[16];
cx q[16],q[2];
cx q[2],q[24];
cx q[31],q[28];
cx q[28],q[2];
cx q[2],q[31];
h q[7];
s q[7];
s q[8];
s q[12];
h q[12];
h q[14];
h q[15];
h q[19];
s q[24];
h q[27];
h q[28];
s q[29];
h q[29];
h q[30];
s q[34];
swap q[11],q[1];
cx q[11],q[4];
cx q[11],q[8];
cx q[11],q[13];
cx q[11],q[24];
cx q[11],q[31];
cx q[11],q[32];
cx q[12],q[11];
cx q[16],q[11];
cx q[19],q[11];
cx q[29],q[11];
cx q[33],q[11];
cx q[1],q[11];
cx q[15],q[27];
cx q[15],q[28];
cx q[15],q[30];
cx q[15],q[34];
cx q[11],q[15];
h q[15];
cx q[15],q[11];
cx q[10],q[7];
cx q[7],q[11];
cx q[11],q[10];
cx q[18],q[14];
cx q[14],q[11];
cx q[11],q[18];
s q[1];
h q[1];
s q[1];
s q[3];
h q[4];
s q[8];
h q[8];
s q[10];
h q[10];
h q[12];
s q[14];
s q[15];
s q[16];
s q[17];
h q[17];
s q[17];
s q[19];
s q[27];
s q[28];
h q[30];
s q[32];
h q[32];
h q[33];
s q[34];
swap q[18],q[1];
cx q[18],q[14];
cx q[18],q[15];
cx q[18],q[27];
cx q[4],q[18];
cx q[7],q[18];
cx q[8],q[18];
cx q[10],q[18];
cx q[12],q[18];
cx q[31],q[18];
cx q[32],q[18];
cx q[3],q[13];
cx q[3],q[19];
cx q[3],q[28];
cx q[3],q[30];
cx q[3],q[33];
cx q[18],q[3];
h q[3];
cx q[3],q[18];
cx q[17],q[16];
cx q[16],q[18];
cx q[18],q[17];
cx q[34],q[24];
cx q[24],q[18];
cx q[18],q[34];
s q[4];
h q[4];
h q[8];
s q[10];
h q[10];
h q[12];
h q[13];
s q[15];
h q[16];
s q[16];
h q[17];
h q[24];
s q[27];
s q[29];
s q[30];
h q[30];
s q[30];
h q[31];
s q[32];
h q[33];
s q[34];
cx q[1],q[14];
cx q[1],q[19];
cx q[1],q[24];
cx q[1],q[28];
cx q[1],q[29];
cx q[1],q[31];
cx q[1],q[32];
cx q[4],q[1];
cx q[8],q[1];
cx q[10],q[1];
cx q[12],q[27];
cx q[12],q[34];
cx q[1],q[12];
h q[12];
cx q[12],q[1];
cx q[15],q[13];
cx q[13],q[1];
cx q[1],q[15];
cx q[17],q[16];
cx q[16],q[1];
cx q[1],q[17];
cx q[33],q[30];
cx q[30],q[1];
cx q[1],q[33];
s q[3];
h q[3];
s q[3];
s q[4];
h q[4];
s q[8];
h q[8];
s q[8];
s q[10];
s q[12];
h q[13];
s q[14];
h q[14];
h q[16];
h q[17];
s q[19];
h q[24];
s q[27];
h q[28];
h q[30];
swap q[31],q[3];
cx q[31],q[24];
cx q[31],q[30];
cx q[4],q[31];
cx q[7],q[31];
cx q[28],q[31];
cx q[34],q[31];
cx q[10],q[12];
cx q[10],q[15];
cx q[10],q[16];
cx q[10],q[17];
cx q[31],q[10];
h q[10];
cx q[10],q[31];
cx q[13],q[8];
cx q[8],q[31];
cx q[31],q[13];
cx q[19],q[14];
cx q[14],q[31];
cx q[31],q[19];
cx q[32],q[27];
cx q[27],q[31];
cx q[31],q[32];
s q[8];
h q[8];
h q[10];
s q[13];
s q[14];
s q[15];
h q[15];
h q[16];
h q[24];
s q[27];
h q[27];
h q[28];
s q[29];
h q[29];
s q[32];
h q[32];
s q[32];
s q[33];
h q[33];
h q[34];
swap q[10],q[3];
cx q[10],q[13];
cx q[10],q[16];
cx q[4],q[10];
cx q[15],q[10];
cx q[19],q[10];
cx q[24],q[10];
cx q[33],q[10];
cx q[7],q[14];
cx q[7],q[28];
cx q[7],q[3];
cx q[10],q[7];
h q[7];
cx q[7],q[10];
cx q[27],q[8];
cx q[8],q[10];
cx q[10],q[27];
cx q[30],q[29];
cx q[29],q[10];
cx q[10],q[30];
cx q[34],q[32];
cx q[32],q[10];
cx q[10],q[34];
s q[3];
h q[3];
s q[3];
h q[4];
s q[8];
h q[12];
h q[13];
s q[15];
h q[15];
h q[17];
s q[19];
h q[27];
s q[27];
s q[28];
h q[28];
h q[29];
s q[30];
s q[34];
swap q[15],q[3];
cx q[15],q[4];
cx q[15],q[7];
cx q[15],q[19];
cx q[15],q[30];
cx q[15],q[34];
cx q[14],q[15];
cx q[29],q[15];
cx q[3],q[15];
cx q[12],q[8];
cx q[8],q[15];
cx q[15],q[12];
cx q[17],q[13];
cx q[13],q[15];
cx q[15],q[17];
cx q[27],q[24];
cx q[24],q[15];
cx q[15],q[27];
cx q[32],q[28];
cx q[28],q[15];
cx q[15],q[32];
h q[3];
h q[8];
h q[12];
s q[12];
h q[13];
h q[16];
s q[27];
h q[28];
h q[29];
s q[30];
h q[30];
s q[30];
s q[33];
h q[33];
swap q[7],q[4];
cx q[7],q[8];
cx q[7],q[17];
cx q[3],q[7];
cx q[16],q[7];
cx q[32],q[7];
cx q[13],q[24];
cx q[13],q[27];
cx q[13],q[29];
cx q[7],q[13];
h q[13];
cx q[13],q[7];
cx q[28],q[12];
cx q[12],q[7];
cx q[7],q[28];
cx q[33],q[30];
cx q[30],q[7];
cx q[7],q[33];
h q[3];
s q[3];
s q[14];
h q[17];
h q[19];
h q[24];
s q[27];
h q[27];
s q[27];
s q[28];
h q[28];
h q[30];
s q[32];
h q[32];
s q[32];
s q[33];
h q[33];
s q[33];
s q[34];
cx q[33],q[12];
cx q[33],q[13];
cx q[33],q[16];
cx q[33],q[34];
cx q[19],q[33];
cx q[28],q[33];
cx q[30],q[33];
cx q[14],q[3];
cx q[3],q[33];
cx q[33],q[14];
cx q[24],q[17];
cx q[17],q[33];
cx q[33],q[24];
cx q[32],q[27];
cx q[27],q[33];
cx q[33],q[32];
h q[4];
s q[4];
h q[8];
h q[13];
h q[14];
s q[16];
h q[17];
h q[27];
s q[28];
h q[28];
s q[28];
s q[32];
cx q[13],q[14];
cx q[13],q[16];
cx q[13],q[17];
cx q[13],q[29];
cx q[13],q[30];
cx q[8],q[13];
cx q[24],q[13];
cx q[27],q[13];
cx q[32],q[34];
cx q[13],q[32];
h q[32];
cx q[32],q[13];
cx q[28],q[4];
cx q[4],q[13];
cx q[13],q[28];
s q[3];
s q[8];
s q[14];
s q[17];
h q[17];
s q[19];
h q[27];
h q[30];
h q[34];
swap q[8],q[3];
cx q[8],q[24];
cx q[8],q[29];
cx q[8],q[34];
cx q[4],q[8];
cx q[17],q[8];
cx q[14],q[19];
cx q[14],q[27];
cx q[14],q[3];
cx q[8],q[14];
h q[14];
cx q[14],q[8];
cx q[30],q[28];
cx q[28],q[8];
cx q[8],q[30];
s q[3];
h q[3];
s q[4];
h q[4];
h q[12];
s q[12];
s q[19];
s q[24];
h q[27];
h q[28];
h q[29];
swap q[28],q[3];
cx q[28],q[17];
cx q[28],q[24];
cx q[28],q[27];
cx q[19],q[29];
cx q[19],q[3];
cx q[28],q[19];
h q[19];
cx q[19],q[28];
cx q[12],q[4];
cx q[4],q[28];
cx q[28],q[12];
cx q[34],q[32];
cx q[32],q[28];
cx q[28],q[34];
h q[3];
s q[14];
h q[14];
h q[17];
s q[17];
s q[19];
h q[19];
s q[19];
h q[24];
s q[30];
h q[30];
swap q[4],q[3];
cx q[4],q[24];
cx q[30],q[4];
cx q[3],q[4];
cx q[12],q[27];
cx q[4],q[12];
h q[12];
cx q[12],q[4];
cx q[16],q[14];
cx q[14],q[4];
cx q[4],q[16];
cx q[19],q[17];
cx q[17],q[4];
cx q[4],q[19];
s q[3];
h q[3];
s q[3];
h q[12];
h q[17];
s q[17];
s q[19];
h q[24];
h q[29];
s q[30];
swap q[30],q[3];
cx q[30],q[12];
cx q[30],q[29];
cx q[24],q[30];
cx q[16],q[3];
cx q[30],q[16];
h q[16];
cx q[16],q[30];
cx q[19],q[17];
cx q[17],q[30];
cx q[30],q[19];
cx q[34],q[32];
cx q[32],q[30];
cx q[30],q[34];
s q[3];
s q[16];
s q[17];
h q[27];
h q[29];
s q[32];
swap q[14],q[3];
cx q[14],q[17];
cx q[14],q[32];
cx q[12],q[14];
cx q[27],q[14];
cx q[29],q[14];
cx q[34],q[14];
cx q[16],q[19];
cx q[16],q[3];
cx q[14],q[16];
h q[16];
cx q[16],q[14];
s q[3];
s q[17];
h q[17];
h q[19];
s q[27];
s q[32];
h q[34];
swap q[24],q[3];
cx q[24],q[27];
cx q[17],q[24];
cx q[19],q[24];
cx q[3],q[24];
cx q[24],q[34];
h q[34];
cx q[34],q[24];
cx q[32],q[29];
cx q[29],q[24];
cx q[24],q[32];
h q[3];
s q[12];
h q[12];
s q[12];
h q[17];
h q[19];
s q[34];
h q[34];
cx q[12],q[3];
cx q[12],q[17];
cx q[12],q[19];
cx q[27],q[12];
cx q[34],q[29];
cx q[29],q[12];
cx q[12],q[34];
h q[3];
s q[16];
h q[16];
s q[16];
h q[17];
h q[34];
cx q[16],q[29];
cx q[3],q[16];
cx q[17],q[16];
cx q[32],q[16];
cx q[16],q[34];
h q[34];
cx q[34],q[16];
s q[3];
h q[17];
h q[19];
s q[27];
h q[27];
s q[27];
s q[29];
s q[32];
h q[32];
s q[32];
s q[34];
cx q[3],q[34];
cx q[17],q[3];
cx q[27],q[19];
cx q[19],q[3];
cx q[3],q[27];
cx q[32],q[29];
cx q[29],q[3];
cx q[3],q[32];
h q[17];
s q[17];
s q[27];
h q[27];
s q[27];
s q[29];
h q[29];
h q[32];
cx q[27],q[19];
cx q[27],q[34];
cx q[32],q[27];
cx q[29],q[17];
cx q[17],q[27];
cx q[27],q[29];
s q[17];
h q[19];
h q[29];
h q[34];
s q[34];
swap q[29],q[17];
cx q[29],q[32];
cx q[17],q[29];
cx q[34],q[19];
cx q[19],q[29];
cx q[29],q[34];
s q[17];
s q[32];
h q[32];
s q[34];
h q[34];
cx q[32],q[17];
cx q[17],q[34];
cx q[34],q[32];
s q[17];
h q[17];
s q[17];
s q[19];
h q[19];
s q[32];
swap q[32],q[17];
cx q[19],q[32];
cx q[32],q[17];
h q[17];
cx q[17],q[32];
h q[17];
s q[17];
h q[19];
swap q[19],q[17];
cx q[19],q[17];
h q[17];
cx q[17],q[19];
s q[17];
y q[0];
y q[1];
z q[4];
x q[5];
y q[7];
y q[8];
z q[9];
y q[10];
x q[11];
z q[12];
x q[13];
y q[14];
x q[15];
x q[16];
x q[17];
y q[18];
z q[19];
y q[20];
z q[22];
y q[23];
z q[24];
x q[25];
x q[27];
y q[29];
x q[30];
x q[31];
y q[33];
y q[34];