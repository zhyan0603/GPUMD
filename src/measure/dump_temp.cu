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
Dump system and subsystem temperatures to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_temp.cuh"
#include "integrate/integrate.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

Dump_Temp::Dump_Temp(const char** param, int num_param, const std::vector<Group>& groups)
{
  parse(param, num_param, groups);
  property_name = "dump_temp";
}

void Dump_Temp::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  if (num_param != 4) {
    PRINT_INPUT_ERROR("dump_temp should have 3 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("temp dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("temp dump interval should > 0.");
  }
  if (strcmp(param[2], "group") != 0) {
    PRINT_INPUT_ERROR("The second option for dump_temp should be 'group'.");
  }
  if (!is_valid_int(param[3], &grouping_method_)) {
    PRINT_INPUT_ERROR("Grouping method for dump_temp should be an integer.");
  }
  if (grouping_method_ < 0 || grouping_method_ >= groups.size()) {
    PRINT_INPUT_ERROR("Grouping method for dump_temp should >= 0 and < number of grouping methods.");
  }
  printf("Dump temperature every %d steps using grouping method %d.\n", dump_interval_, grouping_method_);
}

void Dump_Temp::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  fid_ = my_fopen("temp.out", "a");
  cpu_group_temperature_.resize(group[grouping_method_].number, 0.0);
}

void Dump_Temp::process(
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
  if ((step + 1) % dump_interval_ != 0) {
    return;
  }

  const int number_of_atoms = atom.number_of_atoms;
  const int number_of_groups = group[grouping_method_].number;
  atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());

  for (int group_id = 0; group_id < number_of_groups; ++group_id) {
    const int group_size = group[grouping_method_].cpu_size[group_id];
    const int group_offset = group[grouping_method_].cpu_size_sum[group_id];
    if (group_size <= 0) {
      cpu_group_temperature_[group_id] = 0.0;
      continue;
    }

    double kinetic_energy_twice = 0.0;
    for (int k = 0; k < group_size; ++k) {
      const int atom_id = group[grouping_method_].cpu_contents[group_offset + k];
      const double vx = atom.cpu_velocity_per_atom[atom_id];
      const double vy = atom.cpu_velocity_per_atom[atom_id + number_of_atoms];
      const double vz = atom.cpu_velocity_per_atom[atom_id + number_of_atoms * 2];
      kinetic_energy_twice += atom.cpu_mass[atom_id] * (vx * vx + vy * vy + vz * vz);
    }
    cpu_group_temperature_[group_id] = kinetic_energy_twice / (3.0 * group_size * K_B);
  }

  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);
  const double total_temperature = (integrate.type >= 31) ? temperature_target : thermo[0];

  fprintf(fid_, "%20d%20.10e", step + 1, total_temperature);
  for (int group_id = 0; group_id < number_of_groups; ++group_id) {
    fprintf(fid_, "%20.10e", cpu_group_temperature_[group_id]);
  }
  fprintf(fid_, "\n");
  fflush(fid_);
}

void Dump_Temp::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (fid_ != nullptr) {
    fclose(fid_);
    fid_ = nullptr;
  }
}

