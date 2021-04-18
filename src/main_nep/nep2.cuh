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

#pragma once
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
class Neighbor;

struct NEP2_Data {
  GPU_Vector<int> NN3b;   // 3-body neighbor number
  GPU_Vector<int> NL3b;   // 3-body neighbor list
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
};

class NEP2 : public Potential
{
public:
  struct Para2B {
    float rc = 0.0f;    // cutoff
    float rcinv = 0.0f; // inverse of the cutoff
  };

  struct Para3B {
    float rc = 0.0f;    // cutoff
    float rcinv = 0.0f; // inverse of the cutoff
  };

  struct ParaMB {
    float rc = 0.0f;    // cutoff
    float rcinv = 0.0f; // inverse of the cutoff
    int n_max = 0;      // n = 0, 1, 2, ..., n_max
    int L_max = 0;      // l = 0, 1, 2, ..., L_max
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons_per_layer = 0; // number of neurons per hidden layer
    int num_para = 0;              // number of parameters
    const float* w0;               // weight from the input to the first hidden layer
    const float* b0;               // bias for the first hidden layer
    const float* w1;               // weight from the first to the second hidden layer
    const float* b1;               // bias for the second hidden layer
    const float* w2;               // weight from the second to the output layer
    const float* b2;               // bias for the output layer
  };

  NEP2(
    int num_neurons_2b,
    float rc_2b,
    int num_neurons_3b,
    float rc_3b,
    int num_neurons_mb,
    int n_max,
    int L_max);
  void initialize(int N, int MAX_ATOM_NUMBER);
  void update_potential(const float* parameters);
  void find_force(
    int Nc,
    int N,
    int* Na,
    int* Na_sum,
    int max_Na,
    float* atomic_number,
    float* h,
    Neighbor* neighbor,
    float* r,
    GPU_Vector<float>& f,
    GPU_Vector<float>& virial,
    GPU_Vector<float>& pe);

private:
  Para2B para2b;
  Para3B para3b;
  ParaMB paramb;
  ANN ann2b;
  ANN ann3b;
  ANN annmb;
  NEP2_Data nep_data;
  void update_potential(const float* parameters, ANN& ann);
};