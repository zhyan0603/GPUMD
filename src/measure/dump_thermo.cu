/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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

/*-----------------------------------------------------------------------------------------------100
Dump thermo data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_thermo.cuh"
#include "integrate/integrate.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

Dump_Thermo::Dump_Thermo(const char** param, int num_param) 
{
  parse(param, num_param);
  property_name = "dump_thermo";
}

void Dump_Thermo::parse(const char** param, int num_param)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("dump_thermo should have 1 or 2 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("thermo dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("thermo dump interval should > 0.");
  }
  printf("Dump thermo every %d steps.\n", dump_interval_);

  if (num_param == 3) {
    dump_element_potential_ = true;
    element_symbol_ = param[2];
    if (element_symbol_.empty()) {
      PRINT_INPUT_ERROR("element symbol for dump_thermo should not be empty.");
    }
    if (element_symbol_.size() > 16) {
      PRINT_INPUT_ERROR("element symbol for dump_thermo is too long.");
    }
    for (char c : element_symbol_) {
      if (!std::isalpha(static_cast<unsigned char>(c))) {
        PRINT_INPUT_ERROR("element symbol for dump_thermo should contain only letters.");
      }
    }
    printf("Also dump summed potential energy for element %s.\n", element_symbol_.c_str());
  }
}

void Dump_Thermo::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  fid_ = my_fopen("thermo.out", "a");
  if (dump_element_potential_) {
    element_atom_indices_.clear();
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      if (atom.cpu_atom_symbol[n] == element_symbol_) {
        element_atom_indices_.push_back(n);
      }
    }
    if (element_atom_indices_.empty()) {
      PRINT_INPUT_ERROR("No atoms found for the requested element symbol in dump_thermo.");
    }
    cpu_potential_per_atom_.resize(atom.number_of_atoms);
    std::snprintf(filename_, sizeof(filename_), "thermo_%s.out", element_symbol_.c_str());
    fid_element_ = my_fopen(filename_, "a");
  }
}

void Dump_Thermo::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature_target,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& gpu_thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  int number_of_atoms_fixed =
    (fixed_group < 0) ? 0 : group[integrate.fixed_grouping_method].cpu_size[fixed_group];

  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);
  double energy_kin, temperature;
  if (integrate.type >= 31) {
    energy_kin = thermo[0];
    temperature = temperature_target;
  } else {
    const int number_of_atoms_moving = atom.number_of_atoms - number_of_atoms_fixed;
    energy_kin = 1.5 * number_of_atoms_moving * K_B * thermo[0];
    temperature = thermo[0];
  }

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(
    fid_,
    "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e",
    temperature,
    energy_kin,
    thermo[1],
    thermo[2] * PRESSURE_UNIT_CONVERSION,
    thermo[3] * PRESSURE_UNIT_CONVERSION,
    thermo[4] * PRESSURE_UNIT_CONVERSION,
    thermo[7] * PRESSURE_UNIT_CONVERSION,
    thermo[6] * PRESSURE_UNIT_CONVERSION,
    thermo[5] * PRESSURE_UNIT_CONVERSION);

  fprintf(
    fid_,
    "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e\n",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);
  fflush(fid_);

  if (dump_element_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
    double selected_element_potential = 0.0;
    for (const int atom_index : element_atom_indices_) {
      selected_element_potential += cpu_potential_per_atom_[atom_index];
    }
    fprintf(
      fid_element_,
      "%20d%20.10e%20.10e\n",
      step + 1,
      global_time * TIME_UNIT_CONVERSION,
      selected_element_potential);
    fflush(fid_element_);
  }
}

void Dump_Thermo::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(fid_);
  if (fid_element_ != nullptr) {
    fclose(fid_element_);
    fid_element_ = nullptr;
  }
}
