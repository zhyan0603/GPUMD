/*
    Standalone gpumd_mace executable entry.
*/

#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/read_xyz.cuh"
#include "potentials/mace/mace_potential.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void print_welcome_information()
{
  printf("\n");
  printf("***************************************************************\n");
  printf("*                 Welcome to use GPUMD                        *\n");
  printf("*          This is the gpumd_mace executable                  *\n");
  printf("***************************************************************\n");
  printf("\n");
}

static std::string parse_mace_model_from_runin()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  std::string line;
  while (std::getline(input_run, line)) {
    const std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 3 && tokens[0] == "potential" && tokens[1] == "mace") {
      return tokens[2];
    }
    if (tokens.size() >= 2 && tokens[0] == "potential" && tokens[1] != "mace") {
      return tokens[1];
    }
  }
  PRINT_INPUT_ERROR("Expect line: potential <model.mace> or potential mace <model.mace> in run.in.");
  return "";
}

int main(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  print_welcome_information();
  print_compile_information();
  print_gpu_information();

  print_line_1();
  printf("Started running gpumd_mace.\n");
  print_line_2();

  CHECK(gpuDeviceSynchronize());
  const auto t0 = std::chrono::high_resolution_clock::now();

  Box box;
  Atom atom;
  std::vector<Group> group;
  int has_velocity_in_xyz = 0;
  int number_of_types = 0;
  initialize_position(has_velocity_in_xyz, number_of_types, box, group, atom);
  GPU_Vector<double> thermo;
  allocate_memory_gpu(group, atom, thermo);

  const std::string model_file = parse_mace_model_from_runin();
  mace::Model model;
  mace::load_model(model_file.c_str(), model);

  mace::Workspace ws;
  Neighbor neighbor;
  mace::initialize_neighbor(model.hp, atom.number_of_atoms, neighbor);
  mace::build_local_neighbor(model.hp, box, atom.type, atom.position_per_atom, neighbor, ws);
  mace::compute_inference(
    model,
    box,
    atom.type,
    atom.position_per_atom,
    ws,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom);

  std::vector<double> energy_host(atom.number_of_atoms);
  atom.potential_per_atom.copy_to_host(energy_host.data());
  double e_total = 0.0;
  for (double e : energy_host) {
    e_total += e;
  }

  std::vector<double> force_host((size_t)atom.number_of_atoms * 3);
  atom.force_per_atom.copy_to_host(force_host.data());
  double fnorm2 = 0.0;
  for (double f : force_host) {
    fnorm2 += f * f;
  }

  double stress[9] = {};
  mace::compute_stress_from_virial(box, atom.virial_per_atom, stress);

  CHECK(gpuDeviceSynchronize());
  const auto t1 = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;

  print_line_1();
  printf("MACE inference complete.\n");
  printf("Total potential energy = %.16e\n", e_total);
  printf("||F||_2 = %.16e\n", sqrt(fnorm2));
  printf(
    "Stress (xx yy zz xy xz yz yx zx zy) = %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
    stress[0],
    stress[1],
    stress[2],
    stress[3],
    stress[4],
    stress[5],
    stress[6],
    stress[7],
    stress[8]);
  printf("Time used = %f s.\n", dt.count());
  print_line_2();

  print_line_1();
  printf("Finished running gpumd_mace.\n");
  print_line_2();

  return EXIT_SUCCESS;
}
