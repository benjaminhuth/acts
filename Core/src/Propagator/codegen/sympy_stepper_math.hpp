// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Note: This file is generated by generate_sympy_stepper.py
//       Do not modify it manually.

#pragma once

#include <cmath>

template <typename T, typename GetB>
bool rk4(const T* p, const T* d, const T t, const T h, const T lambda,
         const T m, const T p_abs, GetB getB, T* err, const T errTol, T* new_p,
         T* new_d, T* new_time, T* path_derivatives, T* J) {
  const auto B1 = getB(p);
  const auto x5 = std::pow(h, 2);
  const auto x0 = B1[1] * d[2];
  const auto x1 = B1[0] * d[2];
  const auto x2 = B1[2] * d[0];
  const auto x3 = B1[0] * d[1];
  const auto x4 = h * d[0];
  const auto x7 = h * d[1];
  const auto x8 = h * d[2];
  const auto x6 = (1.0 / 8.0) * x5;
  T k1[3];
  k1[0] = lambda * (-x0 + B1[2] * d[1]);
  k1[1] = lambda * (x1 - x2);
  k1[2] = lambda * (-x3 + B1[1] * d[0]);
  T p2[3];
  p2[0] = (1.0 / 2.0) * x4 + x6 * k1[0] + p[0];
  p2[1] = x6 * k1[1] + (1.0 / 2.0) * x7 + p[1];
  p2[2] = x6 * k1[2] + (1.0 / 2.0) * x8 + p[2];
  const auto B2 = getB(p2);
  const auto x9 = (1.0 / 2.0) * h;
  const auto x19 = (1.0 / 2.0) * x5;
  const auto x11 = lambda * B2[2];
  const auto x13 = lambda * B2[1];
  const auto x15 = lambda * B2[0];
  const auto x20 = x4 + p[0];
  const auto x21 = x7 + p[1];
  const auto x22 = x8 + p[2];
  const auto x10 = x9 * k1[1] + d[1];
  const auto x12 = x9 * k1[2] + d[2];
  const auto x14 = x9 * k1[0] + d[0];
  T k2[3];
  k2[0] = x10 * x11 - x12 * x13;
  k2[1] = lambda * x12 * B2[0] - x11 * x14;
  k2[2] = -x10 * x15 + x13 * x14;
  const auto x16 = x9 * k2[1] + d[1];
  const auto x17 = x9 * k2[2] + d[2];
  const auto x18 = x9 * k2[0] + d[0];
  T k3[3];
  k3[0] = x11 * x16 - x13 * x17;
  k3[1] = lambda * x17 * B2[0] - x11 * x18;
  k3[2] = x13 * x18 - x15 * x16;
  T p3[3];
  p3[0] = x19 * k3[0] + x20;
  p3[1] = x19 * k3[1] + x21;
  p3[2] = x19 * k3[2] + x22;
  const auto B3 = getB(p3);
  const auto x24 = lambda * B3[2];
  const auto x26 = lambda * B3[1];
  const auto x28 = lambda * B3[0];
  const auto x23 = h * k3[1] + d[1];
  const auto x25 = h * k3[2] + d[2];
  const auto x27 = h * k3[0] + d[0];
  T k4[3];
  k4[0] = x23 * x24 - x25 * x26;
  k4[1] = lambda * x25 * B3[0] - x24 * x27;
  k4[2] = -x23 * x28 + x26 * x27;
  const auto x29 = k1[0] + k4[0];
  const auto x30 = k1[1] + k4[1];
  const auto x31 = k1[2] + k4[2];
  *err =
      x5 * (std::fabs(-x29 + k2[0] + k3[0]) + std::fabs(-x30 + k2[1] + k3[1]) +
            std::fabs(-x31 + k2[2] + k3[2]));
  if (*err > errTol) {
    return false;
  }
  const auto x32 = (1.0 / 6.0) * x5;
  new_p[0] = x20 + x32 * (k1[0] + k2[0] + k3[0]);
  new_p[1] = x21 + x32 * (k1[1] + k2[1] + k3[1]);
  new_p[2] = x22 + x32 * (k1[2] + k2[2] + k3[2]);
  const auto x33 = (1.0 / 6.0) * h;
  T new_d_tmp[3];
  new_d_tmp[0] = x33 * (x29 + 2 * k2[0] + 2 * k3[0]) + d[0];
  new_d_tmp[1] = x33 * (x30 + 2 * k2[1] + 2 * k3[1]) + d[1];
  new_d_tmp[2] = x33 * (x31 + 2 * k2[2] + 2 * k3[2]) + d[2];
  const auto x34 = 1.0 / std::sqrt(std::pow(std::fabs(new_d_tmp[0]), 2) +
                                   std::pow(std::fabs(new_d_tmp[1]), 2) +
                                   std::pow(std::fabs(new_d_tmp[2]), 2));
  new_d[0] = x34 * new_d_tmp[0];
  new_d[1] = x34 * new_d_tmp[1];
  new_d[2] = x34 * new_d_tmp[2];
  const auto x35 = std::pow(m, 2);
  const auto dtds = std::sqrt(std::pow(p_abs, 2) + x35) / p_abs;
  *new_time = dtds * h + t;
  if (J == nullptr) {
    return true;
  }
  path_derivatives[0] = new_d[0];
  path_derivatives[1] = new_d[1];
  path_derivatives[2] = new_d[2];
  path_derivatives[3] = dtds;
  path_derivatives[4] = k4[0];
  path_derivatives[5] = k4[1];
  path_derivatives[6] = k4[2];
  path_derivatives[7] = 0;
  const auto x57 = (1.0 / 3.0) * h;
  const auto x36 = std::pow(lambda, 2) * x9;
  const auto x43 = x1 - x2;
  const auto x47 = x11 * x9;
  const auto x48 = x15 * x9;
  const auto x49 = x13 * x9;
  const auto x50 = h * x26;
  const auto x51 = h * x24;
  const auto x52 = h * x28;
  const auto x53 = lambda * x32;
  const auto x58 = lambda * x33;
  const auto x41 = -x3 + B1[1] * d[0];
  const auto x46 = -x0 + B1[2] * d[1];
  const auto x37 = x36 * B2[1];
  const auto x39 = x36 * B2[2];
  const auto x42 = x41 * x9;
  const auto x44 = x36 * B2[0];
  const auto x54 = x53 * B1[2];
  const auto x55 = x53 * B1[1];
  const auto x56 = x53 * B1[0];
  const auto x59 = x58 * B1[2];
  const auto x60 = x58 * B1[1];
  const auto x61 = x58 * B1[0];
  const auto x38 = x37 * B1[1];
  const auto x40 = x39 * B1[2];
  const auto x45 = x44 * B1[0];
  T dk2dTL[12];
  dk2dTL[0] = -x38 - x40;
  dk2dTL[1] = -x11 + x44 * B1[1];
  dk2dTL[2] = x13 + x44 * B1[2];
  dk2dTL[3] = x11 + x37 * B1[0];
  dk2dTL[4] = -x40 - x45;
  dk2dTL[5] = -x15 + x37 * B1[2];
  dk2dTL[6] = -x13 + x39 * B1[0];
  dk2dTL[7] = x15 + x39 * B1[1];
  dk2dTL[8] = -x38 - x45;
  dk2dTL[9] = (1.0 / 2.0) * h * lambda * x43 * B2[2] + x10 * B2[2] -
              x12 * B2[1] - x13 * x42;
  dk2dTL[10] = x12 * B2[0] - x14 * B2[2] + x15 * x42 - x46 * x47;
  dk2dTL[11] = (1.0 / 2.0) * h * lambda * x46 * B2[1] - x10 * B2[0] +
               x14 * B2[1] - x43 * x48;
  T dk3dTL[12];
  dk3dTL[0] = (1.0 / 2.0) * h * lambda * B2[2] * dk2dTL[1] - x49 * dk2dTL[2];
  dk3dTL[1] =
      (1.0 / 2.0) * h * lambda * B2[0] * dk2dTL[2] - x11 - x47 * dk2dTL[0];
  dk3dTL[2] = x13 - x48 * dk2dTL[1] + x49 * dk2dTL[0];
  dk3dTL[3] = x11 + x47 * dk2dTL[4] - x49 * dk2dTL[5];
  dk3dTL[4] = -x47 * dk2dTL[3] + x48 * dk2dTL[5];
  dk3dTL[5] =
      (1.0 / 2.0) * h * lambda * B2[1] * dk2dTL[3] - x15 - x48 * dk2dTL[4];
  dk3dTL[6] =
      (1.0 / 2.0) * h * lambda * B2[2] * dk2dTL[7] - x13 - x49 * dk2dTL[8];
  dk3dTL[7] = x15 - x47 * dk2dTL[6] + x48 * dk2dTL[8];
  dk3dTL[8] = (1.0 / 2.0) * h * lambda * B2[1] * dk2dTL[6] - x48 * dk2dTL[7];
  dk3dTL[9] = (1.0 / 2.0) * h * lambda * B2[2] * dk2dTL[10] + x16 * B2[2] -
              x17 * B2[1] - x49 * dk2dTL[11];
  dk3dTL[10] = x17 * B2[0] - x18 * B2[2] - x47 * dk2dTL[9] + x48 * dk2dTL[11];
  dk3dTL[11] = (1.0 / 2.0) * h * lambda * B2[1] * dk2dTL[9] - x16 * B2[0] +
               x18 * B2[1] - x48 * dk2dTL[10];
  T dFdTL[12];
  dFdTL[0] = h + x32 * dk2dTL[0] + x32 * dk3dTL[0];
  dFdTL[1] = x32 * dk2dTL[1] + x32 * dk3dTL[1] - x54;
  dFdTL[2] = x32 * dk2dTL[2] + x32 * dk3dTL[2] + x55;
  dFdTL[3] = x32 * dk2dTL[3] + x32 * dk3dTL[3] + x54;
  dFdTL[4] = h + x32 * dk2dTL[4] + x32 * dk3dTL[4];
  dFdTL[5] = x32 * dk2dTL[5] + x32 * dk3dTL[5] - x56;
  dFdTL[6] = x32 * dk2dTL[6] + x32 * dk3dTL[6] - x55;
  dFdTL[7] = x32 * dk2dTL[7] + x32 * dk3dTL[7] + x56;
  dFdTL[8] = h + x32 * dk2dTL[8] + x32 * dk3dTL[8];
  dFdTL[9] = x32 * (x46 + dk2dTL[9] + dk3dTL[9]);
  dFdTL[10] = x32 * (x43 + dk2dTL[10] + dk3dTL[10]);
  dFdTL[11] = x32 * (x41 + dk2dTL[11] + dk3dTL[11]);
  T dk4dTL[12];
  dk4dTL[0] = h * lambda * B3[2] * dk3dTL[1] - x50 * dk3dTL[2];
  dk4dTL[1] = h * lambda * B3[0] * dk3dTL[2] - x24 - x51 * dk3dTL[0];
  dk4dTL[2] = x26 + x50 * dk3dTL[0] - x52 * dk3dTL[1];
  dk4dTL[3] = x24 - x50 * dk3dTL[5] + x51 * dk3dTL[4];
  dk4dTL[4] = -x51 * dk3dTL[3] + x52 * dk3dTL[5];
  dk4dTL[5] = h * lambda * B3[1] * dk3dTL[3] - x28 - x52 * dk3dTL[4];
  dk4dTL[6] = h * lambda * B3[2] * dk3dTL[7] - x26 - x50 * dk3dTL[8];
  dk4dTL[7] = x28 - x51 * dk3dTL[6] + x52 * dk3dTL[8];
  dk4dTL[8] = h * lambda * B3[1] * dk3dTL[6] - x52 * dk3dTL[7];
  dk4dTL[9] = h * lambda * B3[2] * dk3dTL[10] + x23 * B3[2] - x25 * B3[1] -
              x50 * dk3dTL[11];
  dk4dTL[10] = x25 * B3[0] - x27 * B3[2] - x51 * dk3dTL[9] + x52 * dk3dTL[11];
  dk4dTL[11] = h * lambda * B3[1] * dk3dTL[9] - x23 * B3[0] + x27 * B3[1] -
               x52 * dk3dTL[10];
  T dGdTL[12];
  dGdTL[0] = x33 * dk4dTL[0] + x57 * dk2dTL[0] + x57 * dk3dTL[0] + 1;
  dGdTL[1] = x33 * dk4dTL[1] + x57 * dk2dTL[1] + x57 * dk3dTL[1] - x59;
  dGdTL[2] = x33 * dk4dTL[2] + x57 * dk2dTL[2] + x57 * dk3dTL[2] + x60;
  dGdTL[3] = x33 * dk4dTL[3] + x57 * dk2dTL[3] + x57 * dk3dTL[3] + x59;
  dGdTL[4] = x33 * dk4dTL[4] + x57 * dk2dTL[4] + x57 * dk3dTL[4] + 1;
  dGdTL[5] = x33 * dk4dTL[5] + x57 * dk2dTL[5] + x57 * dk3dTL[5] - x61;
  dGdTL[6] = x33 * dk4dTL[6] + x57 * dk2dTL[6] + x57 * dk3dTL[6] - x60;
  dGdTL[7] = x33 * dk4dTL[7] + x57 * dk2dTL[7] + x57 * dk3dTL[7] + x61;
  dGdTL[8] = x33 * dk4dTL[8] + x57 * dk2dTL[8] + x57 * dk3dTL[8] + 1;
  dGdTL[9] = x33 * x46 + x33 * dk4dTL[9] + x57 * dk2dTL[9] + x57 * dk3dTL[9];
  dGdTL[10] =
      x33 * x43 + x33 * dk4dTL[10] + x57 * dk2dTL[10] + x57 * dk3dTL[10];
  dGdTL[11] =
      x33 * x41 + x33 * dk4dTL[11] + x57 * dk2dTL[11] + x57 * dk3dTL[11];
  T new_J[64];
  new_J[0] = 1;
  new_J[1] = 0;
  new_J[2] = 0;
  new_J[3] = 0;
  new_J[4] = 0;
  new_J[5] = 0;
  new_J[6] = 0;
  new_J[7] = 0;
  new_J[8] = 0;
  new_J[9] = 1;
  new_J[10] = 0;
  new_J[11] = 0;
  new_J[12] = 0;
  new_J[13] = 0;
  new_J[14] = 0;
  new_J[15] = 0;
  new_J[16] = 0;
  new_J[17] = 0;
  new_J[18] = 1;
  new_J[19] = 0;
  new_J[20] = 0;
  new_J[21] = 0;
  new_J[22] = 0;
  new_J[23] = 0;
  new_J[24] = 0;
  new_J[25] = 0;
  new_J[26] = 0;
  new_J[27] = 1;
  new_J[28] = 0;
  new_J[29] = 0;
  new_J[30] = 0;
  new_J[31] = 0;
  new_J[32] = J[32] * dGdTL[0] + J[40] * dGdTL[1] + J[48] * dGdTL[2] + dFdTL[0];
  new_J[33] = J[33] * dGdTL[0] + J[41] * dGdTL[1] + J[49] * dGdTL[2] + dFdTL[1];
  new_J[34] = J[34] * dGdTL[0] + J[42] * dGdTL[1] + J[50] * dGdTL[2] + dFdTL[2];
  new_J[35] = 0;
  new_J[36] = J[36] * dGdTL[0] + J[44] * dGdTL[1] + J[52] * dGdTL[2];
  new_J[37] = J[37] * dGdTL[0] + J[45] * dGdTL[1] + J[53] * dGdTL[2];
  new_J[38] = J[38] * dGdTL[0] + J[46] * dGdTL[1] + J[54] * dGdTL[2];
  new_J[39] = 0;
  new_J[40] = J[32] * dGdTL[3] + J[40] * dGdTL[4] + J[48] * dGdTL[5] + dFdTL[3];
  new_J[41] = J[33] * dGdTL[3] + J[41] * dGdTL[4] + J[49] * dGdTL[5] + dFdTL[4];
  new_J[42] = J[34] * dGdTL[3] + J[42] * dGdTL[4] + J[50] * dGdTL[5] + dFdTL[5];
  new_J[43] = 0;
  new_J[44] = J[36] * dGdTL[3] + J[44] * dGdTL[4] + J[52] * dGdTL[5];
  new_J[45] = J[37] * dGdTL[3] + J[45] * dGdTL[4] + J[53] * dGdTL[5];
  new_J[46] = J[38] * dGdTL[3] + J[46] * dGdTL[4] + J[54] * dGdTL[5];
  new_J[47] = 0;
  new_J[48] = J[32] * dGdTL[6] + J[40] * dGdTL[7] + J[48] * dGdTL[8] + dFdTL[6];
  new_J[49] = J[33] * dGdTL[6] + J[41] * dGdTL[7] + J[49] * dGdTL[8] + dFdTL[7];
  new_J[50] = J[34] * dGdTL[6] + J[42] * dGdTL[7] + J[50] * dGdTL[8] + dFdTL[8];
  new_J[51] = 0;
  new_J[52] = J[36] * dGdTL[6] + J[44] * dGdTL[7] + J[52] * dGdTL[8];
  new_J[53] = J[37] * dGdTL[6] + J[45] * dGdTL[7] + J[53] * dGdTL[8];
  new_J[54] = J[38] * dGdTL[6] + J[46] * dGdTL[7] + J[54] * dGdTL[8];
  new_J[55] = 0;
  new_J[56] = J[32] * dGdTL[9] + J[40] * dGdTL[10] + J[48] * dGdTL[11] + J[56] +
              dFdTL[9];
  new_J[57] = J[33] * dGdTL[9] + J[41] * dGdTL[10] + J[49] * dGdTL[11] + J[57] +
              dFdTL[10];
  new_J[58] = J[34] * dGdTL[9] + J[42] * dGdTL[10] + J[50] * dGdTL[11] + J[58] +
              dFdTL[11];
  new_J[59] = J[59] + h * lambda * x35 / dtds;
  new_J[60] = J[36] * dGdTL[9] + J[44] * dGdTL[10] + J[52] * dGdTL[11] + J[60];
  new_J[61] = J[37] * dGdTL[9] + J[45] * dGdTL[10] + J[53] * dGdTL[11] + J[61];
  new_J[62] = J[38] * dGdTL[9] + J[46] * dGdTL[10] + J[54] * dGdTL[11] + J[62];
  new_J[63] = 1;
  J[0] = new_J[0];
  J[1] = new_J[1];
  J[2] = new_J[2];
  J[3] = new_J[3];
  J[4] = new_J[4];
  J[5] = new_J[5];
  J[6] = new_J[6];
  J[7] = new_J[7];
  J[8] = new_J[8];
  J[9] = new_J[9];
  J[10] = new_J[10];
  J[11] = new_J[11];
  J[12] = new_J[12];
  J[13] = new_J[13];
  J[14] = new_J[14];
  J[15] = new_J[15];
  J[16] = new_J[16];
  J[17] = new_J[17];
  J[18] = new_J[18];
  J[19] = new_J[19];
  J[20] = new_J[20];
  J[21] = new_J[21];
  J[22] = new_J[22];
  J[23] = new_J[23];
  J[24] = new_J[24];
  J[25] = new_J[25];
  J[26] = new_J[26];
  J[27] = new_J[27];
  J[28] = new_J[28];
  J[29] = new_J[29];
  J[30] = new_J[30];
  J[31] = new_J[31];
  J[32] = new_J[32];
  J[33] = new_J[33];
  J[34] = new_J[34];
  J[35] = new_J[35];
  J[36] = new_J[36];
  J[37] = new_J[37];
  J[38] = new_J[38];
  J[39] = new_J[39];
  J[40] = new_J[40];
  J[41] = new_J[41];
  J[42] = new_J[42];
  J[43] = new_J[43];
  J[44] = new_J[44];
  J[45] = new_J[45];
  J[46] = new_J[46];
  J[47] = new_J[47];
  J[48] = new_J[48];
  J[49] = new_J[49];
  J[50] = new_J[50];
  J[51] = new_J[51];
  J[52] = new_J[52];
  J[53] = new_J[53];
  J[54] = new_J[54];
  J[55] = new_J[55];
  J[56] = new_J[56];
  J[57] = new_J[57];
  J[58] = new_J[58];
  J[59] = new_J[59];
  J[60] = new_J[60];
  J[61] = new_J[61];
  J[62] = new_J[62];
  J[63] = new_J[63];
  return true;
}
