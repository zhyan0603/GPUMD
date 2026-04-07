/*
    Copyright 2026 GPUMD development team
*/

#pragma once

#include "force/neighbor.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdint>

namespace mace
{
static constexpr uint32_t MACE_MAGIC = 0x4D414345u; // "MACE"
static constexpr uint32_t MACE_VERSION = 1u;

struct HyperParameters {
  uint32_t num_species = 0;
  uint32_t num_channels = 0;
  uint32_t num_radial = 0;
  uint32_t num_interactions = 0;
  uint32_t l_max = 0;
  uint32_t max_neighbors = 256;
  float r_max = 0.0f;
  float cutoff_p = 6.0f;
  float cutoff_q = 12.0f;
  float scale = 1.0f;
  float shift = 0.0f;
};

struct Model {
  HyperParameters hp;
  GPU_Vector<float> species_embedding; // [num_species, num_channels]
  GPU_Vector<float> radial_weights;    // [num_interactions, num_channels, num_radial]
  GPU_Vector<float> readout_weight;    // [num_channels]
  GPU_Vector<float> readout_bias;      // [1]
  float readout_bias_scalar = 0.0f;
};

struct Workspace {
  GPU_Vector<int> NN_local;
  GPU_Vector<int> NL_local;
};

void load_model(const char* mace_file, Model& model);
void initialize_neighbor(const HyperParameters& hp, const int num_atoms, Neighbor& neighbor);
void build_local_neighbor(
  const HyperParameters& hp,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  Neighbor& neighbor,
  Workspace& ws);
void compute_inference(
  const Model& model,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const Workspace& ws,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial);
void compute_stress_from_virial(
  const Box& box,
  const GPU_Vector<double>& virial_per_atom,
  double stress_out[9]);
} // namespace mace
