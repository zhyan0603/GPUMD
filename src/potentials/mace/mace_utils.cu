/*
    Helper utilities for standalone MACE potential.
*/

#include "mace_potential.cuh"
#include <algorithm>

namespace mace
{
namespace
{
constexpr int MACE_MIN_NEIGHBOR_CAPACITY = 1024;
}

void initialize_neighbor(const HyperParameters& hp, const int num_atoms, Neighbor& neighbor)
{
  int effective_max_neighbors = (int)hp.max_neighbors;
  // Standalone MACE commonly needs a larger neighbor cap than generic defaults.
  // Keep model-provided value, but enforce a practical lower bound to avoid early overflow.
  effective_max_neighbors = std::max(effective_max_neighbors, MACE_MIN_NEIGHBOR_CAPACITY);
  if (num_atoms > 1) {
    effective_max_neighbors = std::min(effective_max_neighbors, num_atoms - 1);
  } else {
    effective_max_neighbors = 1;
  }
  neighbor.initialize(hp.r_max, num_atoms, effective_max_neighbors);
}

void build_local_neighbor(
  const HyperParameters& hp,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  Neighbor& neighbor,
  Workspace& ws)
{
  const int N = (int)type.size();
  if (N <= 0) {
    return;
  }
  neighbor.find_neighbor_global(hp.r_max, box, type, position);
  if ((int)ws.NN_local.size() != N) {
    ws.NN_local.resize(N);
  }
  const int MN = (int)neighbor.NL.size() / N;
  if ((int)ws.NL_local.size() != N * MN) {
    ws.NL_local.resize(N * MN);
  }
  neighbor.find_local_neighbor_from_global(hp.r_max, box, position, ws.NN_local, ws.NL_local);
}
} // namespace mace
