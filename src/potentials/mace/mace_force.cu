/*
    Virial/stress post-processing for standalone MACE potential.
*/

#include "mace_potential.cuh"
#include <vector>

namespace mace
{
void compute_stress_from_virial(
  const Box& box,
  const GPU_Vector<double>& virial_per_atom,
  double stress_out[9])
{
  for (int i = 0; i < 9; ++i) {
    stress_out[i] = 0.0;
  }

  const int N = (int)virial_per_atom.size() / 9;
  if (N <= 0) {
    return;
  }
  std::vector<double> h_virial((size_t)virial_per_atom.size());
  virial_per_atom.copy_to_host(h_virial.data());

  const double volume = box.get_volume();
  const double inv_volume = volume > 0.0 ? (1.0 / volume) : 0.0;
  for (int c = 0; c < 9; ++c) {
    double sum = 0.0;
    const int offset = c * N;
    for (int i = 0; i < N; ++i) {
      sum += h_virial[offset + i];
    }
    stress_out[c] = -sum * inv_volume;
  }
}
} // namespace mace

