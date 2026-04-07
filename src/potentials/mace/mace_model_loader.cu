/*
    Native MACE binary loader for GPUMD.
*/

#include "mace_potential.cuh"
#include "utilities/error.cuh"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace mace
{
namespace
{
struct FileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t flags;
  uint32_t dtype;
  uint32_t num_species;
  uint32_t num_channels;
  uint32_t num_radial;
  uint32_t num_interactions;
  uint32_t l_max;
  uint32_t max_neighbors;
  float r_max;
  float cutoff_p;
  float cutoff_q;
  float scale;
  float shift;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
};

static std::vector<float> read_tensor_f32(std::ifstream& in)
{
  uint64_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
  if (!in.good()) {
    PRINT_INPUT_ERROR("Failed to read tensor length from .mace file.");
  }
  std::vector<float> v((size_t)count);
  if (count > 0) {
    in.read(reinterpret_cast<char*>(v.data()), sizeof(float) * count);
    if (!in.good()) {
      PRINT_INPUT_ERROR("Failed to read tensor payload from .mace file.");
    }
  }
  return v;
}
} // namespace

void load_model(const char* mace_file, Model& model)
{
  std::ifstream in(mace_file, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "Failed to open " << mace_file << std::endl;
    exit(1);
  }

  FileHeader h{};
  in.read(reinterpret_cast<char*>(&h), sizeof(FileHeader));
  if (!in.good()) {
    PRINT_INPUT_ERROR("Failed to read .mace header.");
  }
  if (h.magic != MACE_MAGIC) {
    PRINT_INPUT_ERROR("Invalid .mace magic number.");
  }
  if (h.version != MACE_VERSION) {
    PRINT_INPUT_ERROR("Unsupported .mace version.");
  }
  if (h.dtype != 0u) {
    PRINT_INPUT_ERROR("Only float32 .mace weights are supported.");
  }

  model.hp.num_species = h.num_species;
  model.hp.num_channels = h.num_channels;
  model.hp.num_radial = h.num_radial;
  model.hp.num_interactions = h.num_interactions;
  model.hp.l_max = h.l_max;
  model.hp.max_neighbors = h.max_neighbors;
  model.hp.r_max = h.r_max;
  model.hp.cutoff_p = h.cutoff_p;
  model.hp.cutoff_q = h.cutoff_q;
  model.hp.scale = h.scale;
  model.hp.shift = h.shift;

  const std::vector<float> species_embedding_h = read_tensor_f32(in);
  const std::vector<float> radial_weights_h = read_tensor_f32(in);
  const std::vector<float> readout_weight_h = read_tensor_f32(in);
  const std::vector<float> readout_bias_h = read_tensor_f32(in);

  const size_t expect_embedding = (size_t)h.num_species * (size_t)h.num_channels;
  const size_t expect_radial =
    (size_t)h.num_interactions * (size_t)h.num_channels * (size_t)h.num_radial;
  const size_t expect_readout = (size_t)h.num_channels;
  if (species_embedding_h.size() != expect_embedding) {
    PRINT_INPUT_ERROR("species_embedding tensor length mismatch.");
  }
  if (radial_weights_h.size() != expect_radial) {
    PRINT_INPUT_ERROR("radial_weights tensor length mismatch.");
  }
  if (readout_weight_h.size() != expect_readout) {
    PRINT_INPUT_ERROR("readout_weight tensor length mismatch.");
  }
  if (readout_bias_h.size() != 1) {
    PRINT_INPUT_ERROR("readout_bias tensor must have length 1.");
  }

  model.species_embedding.resize(species_embedding_h.size());
  model.species_embedding.copy_from_host(species_embedding_h.data());

  model.radial_weights.resize(radial_weights_h.size());
  model.radial_weights.copy_from_host(radial_weights_h.data());

  model.readout_weight.resize(readout_weight_h.size());
  model.readout_weight.copy_from_host(readout_weight_h.data());

  model.readout_bias.resize(1);
  model.readout_bias.copy_from_host(readout_bias_h.data());
  model.readout_bias_scalar = readout_bias_h[0];

  std::cout << "Loaded .mace model:\n";
  std::cout << "  species = " << model.hp.num_species << "\n";
  std::cout << "  channels = " << model.hp.num_channels << "\n";
  std::cout << "  radial basis = " << model.hp.num_radial << "\n";
  std::cout << "  interactions = " << model.hp.num_interactions << "\n";
  std::cout << "  r_max = " << model.hp.r_max << "\n";
}
} // namespace mace
