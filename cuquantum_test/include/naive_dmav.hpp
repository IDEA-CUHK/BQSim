#pragma once

#include "base.hpp"
#include <chrono>

// #include <util/parser.hpp>
using namespace std::chrono;

enum const_nodes {const_zero = -1, const_one = -2};

typedef struct
{
  int qubit;
  int outgoing_DD_edge_ptr[4]; 
} DD_node; // parsed DDNode

typedef struct
{
  cuDoubleComplex w; // weight
  int DD_node_ptr;
} DD_edge; 

// #blocks: rows in mat
// #threads: batchsize
__global__ void naive_dmav(
  DD_edge* dd_edges,
  DD_node* dd_nodes,
  cuDoubleComplex *input_state,
  cuDoubleComplex *output_state,
  int n_qubit,
  int batch_size
) {
  __shared__ int decoded_locs[64];
  __shared__ cuDoubleComplex decoded_factors[64];
  // recording the recursive state of a certain node
  __shared__ uint8_t left_or_right[256]; // left: F right: T
  __shared__ bool up_or_down[256]; // up: F down: T
  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  
  if (tid < 256) {
    left_or_right[tid] = 0;
  }
  __syncthreads();
  // every block decodes the DDNode struct and list the necessary MACs (weights & location) in shared mem
  if (tid == 0) {
    int edge_ptr = 0;
    int node_ptr = 0;
    int stack_ptr = 0;
    int decode_ptr = 0;
    int edge_stack[50];
    edge_stack[stack_ptr] = 0;
    cuDoubleComplex rec_factor = dd_edges[edge_ptr].w;
    int rec_loc = 0;
    // DFS
    while (stack_ptr >= 0) {
      // fetch node
      edge_ptr = edge_stack[stack_ptr];
      node_ptr = dd_edges[edge_ptr].DD_node_ptr;
      if (node_ptr == const_one) {
        decoded_locs[decode_ptr++] = rec_loc + left_or_right[node_ptr];
        decoded_factors[decode_ptr++] = cuCmul(rec_factor, dd_edges[edge_ptr].w);
        stack_ptr--;
        continue;
      }
      else if (node_ptr == const_zero) {
        stack_ptr--;
        continue;
      }
      up_or_down[node_ptr] = bid / (2 << dd_nodes[node_ptr].qubit);
      int child_idx = (int)(left_or_right[node_ptr]) + (int)(up_or_down[node_ptr]) * 2;
      // return or move forward
      if (left_or_right[node_ptr] == 2) {
        rec_factor = cuCdiv(rec_factor, dd_edges[edge_ptr].w);
        rec_loc -= (2 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[node_ptr]-1);
        stack_ptr--;
      }
      else {
        stack_ptr++;
        edge_stack[stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
        left_or_right[node_ptr]++;
        rec_factor = cuCmul(rec_factor, dd_edges[edge_ptr].w);
        rec_loc += (2 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[node_ptr]-1);
      }
      // int child_idx = (int)(left_or_right[tid]) + (int)(up_or_down[tid]) * 2;
      
      // edge_ptr = dd_matrix[node_ptr].DD_node_ptr[child_idx];

    }
  }
  __syncthreads();
  // add 'em up (may be optimized using reduction algo)
  cuDoubleComplex result_value = {0, 0};
  for (int r = 0; r < decode_ptr; r++) {
    cuDoubleComplex temp_value = cuCmul(input_state[decoded_locs[r]*batch_size+tid], decoded_factors[r]);
    result_value = cuCadd(result_value, temp_value);
  }
  __syncthreads();
  output_state[bid*batch_size +tid] = result_value;
}