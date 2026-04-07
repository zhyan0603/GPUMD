/*
    Helper utilities for standalone MACE potential.
*/

#include "mace_potential.cuh"

namespace mace
{
void initialize_neighbor(const HyperParameters& hp, const int num_atoms, Neighbor& neighbor)
{
  neighbor.initialize(hp.r_max, num_atoms, (int)hp.max_neighbors);
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

