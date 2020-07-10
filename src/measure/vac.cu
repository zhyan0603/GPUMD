/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Calculate:
    Velocity AutoCorrelation (VAC) function
    Self Diffusion Coefficient (SDC)
    Mass-weighted VAC (MVAC)
    Phonon Density Of States (PDOS or simply DOS)

Reference for PDOS:
    J. M. Dickey and A. Paskin,
    Computer Simulation of the Lattice Dynamics of Solids,
    Phys. Rev. 188, 1407 (1969).
------------------------------------------------------------------------------*/

#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "vac.cuh"

const int BLOCK_SIZE = 128;

static __global__ void gpu_copy_mass(
  const int N,
  const int offset,
  const int* g_group_contents,
  double* g_mass_o,
  const double* g_mass_i)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < N) {
    int m = g_group_contents[offset + n];
    g_mass_o[n] = g_mass_i[m];
  }
}

void VAC::preprocess(
  const double time_step, const std::vector<Group>& groups, const GPU_Vector<double>& mass)
{
  if (!compute_dos && !compute_sdc)
    return;

  if (compute_dos == compute_sdc) {
    PRINT_INPUT_ERROR("Cannot calculate DOS and SDC simultaneously.");
  }

  dt = time_step * sample_interval;
  dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // natural to ps

  // initialize the number of time origins
  num_time_origins = 0;

  // determine the number of atoms in the selected group
  if (-1 == grouping_method) {
    N = mass.size();
  } else {
    N = groups[grouping_method].cpu_size[group];
  }

  // only need to record Nc frames of velocity data (saving a lot of memory)
  vx.resize(N * Nc);
  vy.resize(N * Nc);
  vz.resize(N * Nc);

  // using unified memory for VAC and initializing to zero
  vac_x.resize(Nc, 0.0, Memory_Type::managed);
  vac_y.resize(Nc, 0.0, Memory_Type::managed);
  vac_z.resize(Nc, 0.0, Memory_Type::managed);

  if (compute_dos) {
    // set default number of DOS points
    if (num_dos_points == -1) {
      num_dos_points = Nc;
    }

    // check if the sampling frequency is large enough
    double nu_max = 1000.0 / (time_step * sample_interval); // THz
    if (nu_max < omega_max / PI) {
      PRINT_INPUT_ERROR("VAC sampling rate < Nyquist frequency.");
    }

    // need mass for DOS calculations
    mass_.resize(N);

    if (grouping_method >= 0) {
      gpu_copy_mass<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        N, groups[grouping_method].cpu_size_sum[group], groups[grouping_method].contents.data(),
        mass_.data(), mass.data());
      CUDA_CHECK_KERNEL
    } else {
      mass_.copy_from_device(mass.data());
    }
  }
}

static __global__ void gpu_copy_velocity(
  const int N,
  const int offset,
  const int* g_group_contents,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < N) {
    int m = g_group_contents[offset + n];
    g_vx_o[n] = g_vx_i[m];
    g_vy_o[n] = g_vy_i[m];
    g_vz_o[n] = g_vz_i[m];
  }
}

static __global__ void gpu_copy_velocity(
  const int N,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < N) {
    g_vx_o[n] = g_vx_i[n];
    g_vy_o[n] = g_vy_i[n];
    g_vz_o[n] = g_vz_i[n];
  }
}

static __global__ void gpu_find_vac(
  const int N,
  const int correlation_step,
  const int compute_dos,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_vx_all,
  const double* g_vy_all,
  const double* g_vz_all,
  double* g_vac_x,
  double* g_vac_y,
  double* g_vac_z)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_sum = bid * N;
  int number_of_rounds = (N - 1) / BLOCK_SIZE + 1;
  __shared__ double s_vac_x[BLOCK_SIZE];
  __shared__ double s_vac_y[BLOCK_SIZE];
  __shared__ double s_vac_z[BLOCK_SIZE];
  double vac_x = 0.0;
  double vac_y = 0.0;
  double vac_z = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * BLOCK_SIZE;
    if (n < N) {
      double mass = compute_dos ? g_mass[n] : 1.0;
      vac_x += mass * g_vx[n] * g_vx_all[size_sum + n];
      vac_y += mass * g_vy[n] * g_vy_all[size_sum + n];
      vac_z += mass * g_vz[n] * g_vz_all[size_sum + n];
    }
  }
  s_vac_x[tid] = vac_x;
  s_vac_y[tid] = vac_y;
  s_vac_z[tid] = vac_z;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_vac_x[tid] += s_vac_x[tid + offset];
      s_vac_y[tid] += s_vac_y[tid + offset];
      s_vac_z[tid] += s_vac_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (bid <= correlation_step) {
      g_vac_x[correlation_step - bid] += s_vac_x[0];
      g_vac_y[correlation_step - bid] += s_vac_y[0];
      g_vac_z[correlation_step - bid] += s_vac_z[0];
    } else {
      g_vac_x[correlation_step + gridDim.x - bid] += s_vac_x[0];
      g_vac_y[correlation_step + gridDim.x - bid] += s_vac_y[0];
      g_vac_z[correlation_step + gridDim.x - bid] += s_vac_z[0];
    }
  }
}

void VAC::process(
  const int step, const std::vector<Group>& groups, const GPU_Vector<double>& velocity_per_atom)
{
  if (!(compute_dos || compute_sdc))
    return;
  if ((step + 1) % sample_interval != 0) {
    return;
  }
  int sample_step = step / sample_interval; // 0, 1, ..., Nc-1, Nc, Nc+1, ...
  int correlation_step = sample_step % Nc;  // 0, 1, ..., Nc-1, 0, 1, ...
  int offset = correlation_step * N;

  const int number_of_atoms_total = velocity_per_atom.size() / 3;

  // copy the velocity data at the current step to appropriate place
  if (grouping_method >= 0) {
    gpu_copy_velocity<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
      N, groups[grouping_method].cpu_size_sum[group], groups[grouping_method].contents.data(),
      vx.data() + offset, vy.data() + offset, vz.data() + offset, velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total);
  } else {
    gpu_copy_velocity<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
      N, vx.data() + offset, vy.data() + offset, vz.data() + offset, velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total);
  }
  CUDA_CHECK_KERNEL

  // start to calculate the VAC (or MVAC) when we have enough frames
  if (sample_step >= Nc - 1) {
    ++num_time_origins;

    gpu_find_vac<<<Nc, BLOCK_SIZE>>>(
      N, correlation_step, compute_dos, mass_.data(), vx.data() + offset, vy.data() + offset,
      vz.data() + offset, vx.data(), vy.data(), vz.data(), vac_x.data(), vac_y.data(),
      vac_z.data());
    CUDA_CHECK_KERNEL
  }
}

static void perform_dft(
  const int N,
  const int Nc,
  const int num_dos_points,
  const double delta_t,
  const double omega_0,
  const double d_omega,
  double* vac_x,
  double* vac_y,
  double* vac_z,
  double* dos_x,
  double* dos_y,
  double* dos_z)
{
  // Apply Hann window and normalize by the correct factor
  for (int nc = 0; nc < Nc; nc++) {
    double hann_window = (cos((PI * nc) / Nc) + 1.0) * 0.5;

    double multiply_factor = 2.0 * hann_window;
    if (nc == 0) {
      multiply_factor = 1.0 * hann_window;
    }

    vac_x[nc] *= multiply_factor;
    vac_y[nc] *= multiply_factor;
    vac_z[nc] *= multiply_factor;
  }

  // Calculate DOS by discrete Fourier transform
  for (int nw = 0; nw < num_dos_points; nw++) {
    double omega = omega_0 + nw * d_omega;
    for (int nc = 0; nc < Nc; nc++) {
      double cos_factor = cos(omega * nc * delta_t);
      dos_x[nw] += vac_x[nc] * cos_factor;
      dos_y[nw] += vac_y[nc] * cos_factor;
      dos_z[nw] += vac_z[nc] * cos_factor;
    }
    dos_x[nw] *= delta_t * 2.0 * N;
    dos_y[nw] *= delta_t * 2.0 * N;
    dos_z[nw] *= delta_t * 2.0 * N;
  }
}

void VAC::find_dos(const char* input_dir)
{
  double d_omega = omega_max / num_dos_points;
  double omega_0 = d_omega;

  // initialize DOS data
  std::vector<double> dos_x(num_dos_points, 0.0);
  std::vector<double> dos_y(num_dos_points, 0.0);
  std::vector<double> dos_z(num_dos_points, 0.0);

  // perform DFT to get DOS from normalized MVAC
  perform_dft(
    N, Nc, num_dos_points, dt_in_ps, omega_0, d_omega, vac_x.data(), vac_y.data(), vac_z.data(),
    dos_x.data(), dos_y.data(), dos_z.data());

  // output DOS
  char file_dos[200];
  strcpy(file_dos, input_dir);
  strcat(file_dos, "/dos.out");
  FILE* fid = fopen(file_dos, "a");
  for (int nw = 0; nw < num_dos_points; nw++) {
    double omega = omega_0 + d_omega * nw;
    fprintf(fid, "%g %g %g %g\n", omega, dos_x[nw], dos_y[nw], dos_z[nw]);
  }
  fflush(fid);
  fclose(fid);
}

static void integrate_vac(
  const int Nc,
  const double dt,
  const double* vac_x,
  const double* vac_y,
  const double* vac_z,
  double* sdc_x,
  double* sdc_y,
  double* sdc_z)
{
  double dt2 = dt * 0.5;
  for (int nc = 1; nc < Nc; nc++) {
    sdc_x[nc] = sdc_x[nc - 1] + (vac_x[nc - 1] + vac_x[nc]) * dt2;
    sdc_y[nc] = sdc_y[nc - 1] + (vac_y[nc - 1] + vac_y[nc]) * dt2;
    sdc_z[nc] = sdc_z[nc - 1] + (vac_z[nc - 1] + vac_z[nc]) * dt2;
  }
}

void VAC::find_sdc(const char* input_dir)
{
  // initialize the SDC data
  std::vector<double> sdc_x(Nc, 0.0);
  std::vector<double> sdc_y(Nc, 0.0);
  std::vector<double> sdc_z(Nc, 0.0);

  // get the SDC from the VAC according to the Green-Kubo relation
  integrate_vac(
    Nc, dt, vac_x.data(), vac_y.data(), vac_z.data(), sdc_x.data(), sdc_y.data(), sdc_z.data());

  // output the VAC and SDC
  char file_sdc[200];
  strcpy(file_sdc, input_dir);
  strcat(file_sdc, "/sdc.out");
  FILE* fid = fopen(file_sdc, "a");
  for (int nc = 0; nc < Nc; nc++) {
    double t = nc * dt_in_ps;

    // change to A^2/ps^2
    vac_x[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;
    vac_y[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;
    vac_z[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;

    sdc_x[nc] *= 1000.0 / TIME_UNIT_CONVERSION; // change to A^2/ps
    sdc_y[nc] *= 1000.0 / TIME_UNIT_CONVERSION; // change to A^2/ps
    sdc_z[nc] *= 1000.0 / TIME_UNIT_CONVERSION; // change to A^2/ps

    fprintf(fid, "%g %g %g %g ", t, vac_x[nc], vac_y[nc], vac_z[nc]);
    fprintf(fid, "%g %g %g\n", sdc_x[nc], sdc_y[nc], sdc_z[nc]);
  }
  fflush(fid);
  fclose(fid);
}

void VAC::postprocess(const char* input_dir)
{
  if (!(compute_dos || compute_sdc))
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU

  // calculate DOS or SDC
  if (compute_dos) {
    // normalize to vac_x[0] = vac_y[0] = vac_z[0] = 1
    double vac_x_0 = vac_x[0];
    double vac_y_0 = vac_y[0];
    double vac_z_0 = vac_z[0];
    for (int nc = 0; nc < Nc; nc++) {
      vac_x[nc] /= vac_x_0;
      vac_y[nc] /= vac_y_0;
      vac_z[nc] /= vac_z_0;
    }

    // output normalized MVAC
    char file_vac[200];
    strcpy(file_vac, input_dir);
    strcat(file_vac, "/mvac.out");
    FILE* fid = fopen(file_vac, "a");
    for (int nc = 0; nc < Nc; nc++) {
      double t = nc * dt_in_ps;
      fprintf(fid, "%g %g %g %g\n", t, vac_x[nc], vac_y[nc], vac_z[nc]);
    }
    fflush(fid);
    fclose(fid);

    // calculate and output DOS
    find_dos(input_dir);
  } else {
    // normalize by the number of atoms and number of time origins
    for (int nc = 0; nc < Nc; nc++) {
      vac_x[nc] /= double(N) * num_time_origins;
      vac_y[nc] /= double(N) * num_time_origins;
      vac_z[nc] /= double(N) * num_time_origins;
    }
    find_sdc(input_dir);
  }

  compute_dos = 0;
  compute_sdc = 0;
  grouping_method = -1;
  group = -1;
  num_dos_points = -1;
}

// Helper functions for parse_compute_dos
void VAC::parse_group(char** param, int* k, Group* groups)
{
  // grouping_method
  if (!is_valid_int(param[*k + 1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method for VAC should be ans integer number.\n");
  }
  if (grouping_method < 0 || grouping_method > 2) {
    PRINT_INPUT_ERROR("grouping method for VAC should be 0 <= x <= 2.\n");
  }
  // group
  if (!is_valid_int(param[*k + 2], &group)) {
    PRINT_INPUT_ERROR("group for VAC should be an integer number.\n");
  }
  if (group < 0 || group > groups[grouping_method].number) {
    PRINT_INPUT_ERROR("group for VAC must be >= 0 and < number of groups.\n");
  }
  *k += 2; // update index for next command
}

void VAC::parse_num_dos_points(char** param, int* k)
{
  // number of DOS points
  if (!is_valid_int(param[*k + 1], &num_dos_points)) {
    PRINT_INPUT_ERROR("number of DOS points for VAC should be an integer "
                      "number.\n");
  }
  if (num_dos_points < 1) {
    PRINT_INPUT_ERROR("number of DOS points for DOS must be > 0.\n");
  }
  *k += 1; //
}

void VAC::parse_compute_dos(char** param, int num_param, Group* groups)
{
  printf("Compute phonon DOS.\n");
  compute_dos = 1;

  if (num_param < 4) {
    PRINT_INPUT_ERROR("compute_dos should have at least 3 parameters.\n");
  }
  if (num_param > 9) {
    PRINT_INPUT_ERROR("compute_dos has too many parameters.\n");
  }

  // sample interval
  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for VAC should be an integer number.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("sample interval for VAC should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  // number of correlation steps
  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for VAC should be an integer number.\n");
  }
  if (Nc <= 0) {
    PRINT_INPUT_ERROR("Nc for VAC should be positive.\n");
  }
  printf("    Nc is %d.\n", Nc);

  // maximal omega
  if (!is_valid_real(param[3], &omega_max)) {
    PRINT_INPUT_ERROR("omega_max should be a real number.\n");
  }
  if (omega_max <= 0) {
    PRINT_INPUT_ERROR("omega_max should be positive.\n");
  }
  printf("    omega_max is %g THz.\n", omega_max);

  // Process optional arguments
  for (int k = 4; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      // check if there are enough inputs
      if (k + 3 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for optional "
                          "'group' DOS command.\n");
      }
      parse_group(param, &k, groups);
      printf("    grouping_method is %d and group is %d.\n", grouping_method, group);
    } else if (strcmp(param[k], "num_dos_points") == 0) {
      // check if there are enough inputs
      if (k + 2 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for optional "
                          "'group' dos command.\n");
      }
      parse_num_dos_points(param, &k);
      printf("    num_dos_points is %d.\n", num_dos_points);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_dos.\n");
    }
  }
}

void VAC::parse_compute_sdc(char** param, int num_param, Group* groups)
{
  printf("Compute SDC.\n");
  compute_sdc = 1;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("compute_sdc should have at least 2 parameters.\n");
  }
  if (num_param > 6) {
    PRINT_INPUT_ERROR("compute_sdc has too many parameters.\n");
  }

  // sample interval
  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for VAC should be an integer number.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("sample interval for VAC should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  // number of correlation steps
  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for VAC should be an integer number.\n");
  }
  if (Nc <= 0) {
    PRINT_INPUT_ERROR("Nc for VAC should be positive.\n");
  }
  printf("    Nc is %d.\n", Nc);

  // Process optional arguments
  for (int k = 3; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      // check if there are enough inputs
      if (k + 3 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for optional "
                          "'group' SDC command.\n");
      }
      parse_group(param, &k, groups);
      printf("    grouping_method is %d and group is %d.\n", grouping_method, group);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_sdc.\n");
    }
  }
}