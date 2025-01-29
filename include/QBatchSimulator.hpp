#ifndef QBATCH_SIMULATOR_H
#define QBATCH_SIMULATOR_H



#include "QuantumComputation.hpp"
#include "Definitions.hpp"
#include "dd/Package.hpp"
#include "operations/OpType.hpp"
#include "dd/Export.hpp"
#include "dd/Operations.hpp"
#include "CircuitOptimizer.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

enum DDELLConversion
{   
  DDELL_GPU = 0, 
  DDELL_CPU = 1, 
  DDELL_Mixed = 2
};


// #define ONE_GB 1*1024*1024

__global__ void replicate(cuDoubleComplex *input_arr_d, int N) {
  input_arr_d[threadIdx.x+blockIdx.x*N] = input_arr_d[blockIdx.x*N];
}

__global__ void initial_check(cuDoubleComplex *input_arr_d, bool *identical, int N) {
  extern __shared__ bool s[];
  __shared__ int res[1];
  if (threadIdx.x == 0) {
    res[0] = true;
  }
  __syncthreads();
  s[threadIdx.x] = ((input_arr_d[threadIdx.x+blockIdx.x*N].x == input_arr_d[blockIdx.x*N].x) && 
    (input_arr_d[threadIdx.x+blockIdx.x*N].y == input_arr_d[blockIdx.x*N].y));
  __syncthreads();
  atomicAnd(res, (int)s[threadIdx.x]);
  __syncthreads();
  if (threadIdx.x == 0) {
    identical[blockIdx.x] = res[0];
  }
}


__global__ void dd_extract_matrix(
  dd::GPU_DD_edge* dd_edges,
  dd::GPU_DD_node* dd_nodes,
  cuDoubleComplex *fused_gate_val,
  int *fused_gate_indices,
  int num_nodes,
  int num_edges,
  int num_non_zeros,
  int num_qubits
) {
  __shared__ int decoded_locs[MAX_DECODED_MACS];
  __shared__ cuDoubleComplex decoded_factors[MAX_DECODED_MACS];
  // recording the recursive state of a certain node
  __shared__ uint8_t left_or_right[MAX_LEV]; // left: F right: T
  __shared__ bool up_or_down[MAX_LEV]; // up: F down: T
  __shared__ int decode_ptr[1];
  __shared__ int edge_stack[MAX_LEV];

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  
  if (tid < num_qubits) {
    left_or_right[tid] = 0;
    up_or_down[num_qubits-1-tid] = bid & (1 << tid);
  }
  __syncthreads();

  // every block decodes the DDNode struct and list the necessary MACs (weights & location) in shared mem
  if (tid == 0) {
    int edge_ptr = 0;
    int node_ptr = 0;
    int stack_ptr = 0;
    decode_ptr[0] = 0;
    
    edge_stack[stack_ptr] = 0;
    cuDoubleComplex rec_factor = {1, 0};
    int rec_loc = 0; // recursive location
    // DFS
    while (stack_ptr >= 0) {
      if (decode_ptr[0] == num_non_zeros) break;
      // fetch node
      edge_ptr = edge_stack[stack_ptr];
      if (edge_ptr == dd::const_zero_edge) {
        stack_ptr--;
        continue;
      }
      node_ptr = dd_edges[edge_ptr].DD_node_ptr;
      if (node_ptr == dd::const_one_node) {
        decoded_locs[decode_ptr[0]] = rec_loc;
        decoded_factors[decode_ptr[0]] = cuCmul(rec_factor, dd_edges[edge_ptr].w);
        stack_ptr--; decode_ptr[0]++;
        continue;
      }

      int child_idx = (int)(left_or_right[stack_ptr]) + (int)(up_or_down[stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[stack_ptr] == 2) {
        left_or_right[stack_ptr] = 0;
        rec_factor = cuCdiv(rec_factor, dd_edges[edge_ptr].w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[stack_ptr]++;
        rec_factor = (left_or_right[stack_ptr] == 1)? cuCmul(rec_factor, dd_edges[edge_ptr].w) : rec_factor;
        rec_loc += (1 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[stack_ptr] -1);
        stack_ptr++;
        edge_stack[stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncthreads();
  if (tid < num_non_zeros) {
    fused_gate_val[bid * num_non_zeros + tid] = {0, 0};
    fused_gate_indices[bid * num_non_zeros + tid] = 0;
  }
  __syncthreads();

  if (tid < decode_ptr[0]) {
    fused_gate_val[bid * num_non_zeros + tid] = decoded_factors[tid];
    fused_gate_indices[bid * num_non_zeros + tid] = decoded_locs[tid];
  }
  __syncthreads();

}

__global__ void dd_extract_matrix_warp(
  dd::GPU_DD_edge* dd_edges,
  dd::GPU_DD_node* dd_nodes,
  cuDoubleComplex *fused_gate_val,
  int *fused_gate_indices,
  int num_nodes,
  int num_edges,
  int num_non_zeros,
  int num_qubits
) {
  __shared__ int decoded_locs[MAX_DECODED_MACS*WARPS_PER_BLOCK];
  __shared__ cuDoubleComplex decoded_factors[MAX_DECODED_MACS*WARPS_PER_BLOCK];
  // recording the recursive state of a certain node
  __shared__ uint8_t left_or_right[MAX_LEV*WARPS_PER_BLOCK]; // left: F right: T
  __shared__ bool up_or_down[MAX_LEV*WARPS_PER_BLOCK]; // up: F down: T
  __shared__ int decode_ptr[WARPS_PER_BLOCK];
  __shared__ int edge_stack[MAX_LEV*WARPS_PER_BLOCK];

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  
  if (tid%WARP_SIZE < num_qubits) {
    left_or_right[MAX_LEV*(tid/WARP_SIZE) + tid%WARP_SIZE] = 0;
    up_or_down[MAX_LEV*(tid/WARP_SIZE) + num_qubits-1-tid%WARP_SIZE] = (bid*WARPS_PER_BLOCK+tid/WARP_SIZE) & (1 << tid%WARP_SIZE);
  }
  __syncwarp();

  // every block decodes the DDNode struct and list the necessary MACs (weights & location) in shared mem
  if (tid%WARP_SIZE == 0) {
    int edge_ptr = 0;
    int node_ptr = 0;
    int stack_ptr = 0;
    decode_ptr[tid/WARP_SIZE] = 0;
    
    edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr] = 0;
    cuDoubleComplex rec_factor = {1, 0};
    int rec_loc = 0; // recursive location
    // DFS
    while (stack_ptr >= 0) {
      if (decode_ptr[tid/WARP_SIZE] == num_non_zeros) break;
      // fetch node
      edge_ptr = edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr];
      if (edge_ptr == dd::const_zero_edge) {
        stack_ptr--;
        continue;
      }
      node_ptr = dd_edges[edge_ptr].DD_node_ptr;
      if (node_ptr == dd::const_one_node) {
        decoded_locs[MAX_DECODED_MACS*(tid/WARP_SIZE)+decode_ptr[tid/WARP_SIZE]] = rec_loc;
        decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+decode_ptr[tid/WARP_SIZE]] = cuCmul(rec_factor, dd_edges[edge_ptr].w);
        stack_ptr--; decode_ptr[tid/WARP_SIZE]++;
        continue;
      }

      int child_idx = (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) + (int)(up_or_down[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 2) {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] = 0;
        rec_factor = cuCdiv(rec_factor, dd_edges[edge_ptr].w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]++;
        rec_factor = (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 1)? cuCmul(rec_factor, dd_edges[edge_ptr].w) : rec_factor;
        rec_loc += (1 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] -1);
        stack_ptr++;
        edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncwarp();
  if (tid%WARP_SIZE < num_non_zeros) {
    fused_gate_val[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = {0, 0};
    fused_gate_indices[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = 0;
  }
  __syncwarp();

  if (tid%WARP_SIZE < decode_ptr[tid/WARP_SIZE]) {
    fused_gate_val[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+tid%WARP_SIZE];
    fused_gate_indices[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = decoded_locs[MAX_DECODED_MACS*(tid/WARP_SIZE)+tid%WARP_SIZE];
  }
  __syncwarp();

}



// __global__ void run_fused_gate(
//   cuDoubleComplex *gates_val,
//   int *gates_indices,
//   int num_non_zero,
//   cuDoubleComplex *input_state,
//   cuDoubleComplex *output_state,
//   int batch_size, 
//   int nDim
// ) {
  
//   int tidx = threadIdx.x;
//   int tidy = threadIdx.y;
//   int tid = tidx+tidy*blockDim.y;
//   int rounds = nDim / (gridDim.x*blockDim.y);
//   int bid = blockIdx.x;
//   __shared__ int share_indices[MAX_VAL];
//   __shared__ cuDoubleComplex shared_val[MAX_VAL];


//   for (int i = 0; i < rounds; i++) {
//     if (tid < num_non_zero * blockDim.y) {
//       share_indices[tid] = gates_indices[((rounds * bid+i)*blockDim.y) * num_non_zero + tid];
//       shared_val[tid] = gates_val[((rounds * bid+i)*blockDim.y) * num_non_zero + tid];
//     }
//     __syncthreads();
//     cuDoubleComplex result_value = {0, 0};
//     for (int j = 0; j < num_non_zero; j++) {
//       cuDoubleComplex temp_value = cuCmul(input_state[share_indices[tidy*num_non_zero+j]*batch_size+tidx], shared_val[tidy*num_non_zero+j]);
//       result_value = cuCadd(result_value, temp_value);
//     }
//     __syncthreads();
//     output_state[((rounds * bid+i)*blockDim.y+tidy)*batch_size +tidx] = result_value;
//   }
//   __syncthreads();
// }

__global__ void run_fused_gate(
  cuDoubleComplex *gates_val,
  int *gates_indices,
  int num_non_zero,
  cuDoubleComplex *input_state,
  cuDoubleComplex *output_state,
  int batch_size, 
  int nDim
) {
  int rows = nDim / gridDim.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ int share_indices[MAX_DECODED_MACS];
  __shared__ cuDoubleComplex shared_val[MAX_DECODED_MACS];

  // for (int idx = tid; idx < num_non_zero * rows; idx += blockDim.x) {
  //   share_indices[idx] = gates_indices[rows * bid * num_non_zero + idx];
  //   shared_val[idx] = gates_val[rows * bid * num_non_zero + idx];
  // }
  // __syncthreads();

  for (int i = 0; i < rows; i++) {
    for (int idx = tid; idx < num_non_zero; idx += blockDim.x) {
      share_indices[idx] = gates_indices[rows * bid * num_non_zero + i * num_non_zero+idx];
      shared_val[idx] = gates_val[rows * bid * num_non_zero + i * num_non_zero+idx];
    }
    __syncthreads();

    cuDoubleComplex result_value = {0, 0};
    for (int j = 0; j < num_non_zero; j++) {
      // cuDoubleComplex temp_value = cuCmul(input_state[share_indices[i*num_non_zero+j]*batch_size+tid], shared_val[i*num_non_zero+j]);
      cuDoubleComplex temp_value = cuCmul(input_state[share_indices[j]*batch_size+tid], shared_val[j]);
      result_value = cuCadd(result_value, temp_value);
    }
    __syncthreads();
    output_state[(rows * bid +i)*batch_size +tid] = result_value;
  }
  __syncthreads();
}


template<class Config = dd::DDPackageConfig>
class QBatchSimulator {
public:
    explicit QBatchSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_, int batch_size_, int num_batch_) : 
    qc(std::move(qc_)), batch_size(batch_size_), num_batch(num_batch_) 
    {
        QBatchSimulator<Config>::dd->resize(qc->getNqubits());
        const auto nQubits = qc->getNqubits();
        nDim    = std::pow(2, nQubits);
        
        cuDoubleComplex *h_batch0;
        cuDoubleComplex *h_batch1;
        checkCudaErrors(cudaMallocHost((void**)&h_batch0, nDim * batch_size_ * sizeof(cuDoubleComplex)));
        checkCudaErrors(cudaMallocHost((void**)&h_batch1, nDim * batch_size_ * sizeof(cuDoubleComplex)));

        std::string filename = "../../input_batch/n"+std::to_string(nQubits)+".txt";
        std::ifstream file;
        file.open((filename).c_str());

        if (!file.is_open()) {
            std::cerr << "Failed to open file." << std::endl;
            exit(-1);
        }
        std::string line;
        while (getline(file, line)) {
            std::istringstream iss(line);
            double real, imag;
            int amp_id = 0;
            while (iss >> real >> imag) {
            h_batch0[amp_id*batch_size_] = {real, imag};
            amp_id++;
            }
        }
        file.close();

        cuDoubleComplex *input_d;
        checkCudaErrors(cudaMalloc((void**)&input_d, nDim * batch_size_ * sizeof(cuDoubleComplex)));
        checkCudaErrors(cudaMemcpy(input_d, h_batch0, nDim * batch_size_ * sizeof(cuDoubleComplex),
                cudaMemcpyHostToDevice));
        replicate<<<nDim, batch_size>>>(input_d, batch_size_);
        checkCudaErrors(cudaMemcpy(h_batch0, input_d, nDim * batch_size_ * sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(input_d));
        
        memset(h_batch1, 0, nDim * batch_size_ * sizeof(cuDoubleComplex));
        h_batch.push_back(h_batch0);
        h_batch.push_back(h_batch1);

        for (int buf = 0; buf < 4; buf++) {
          cuDoubleComplex *d_batch_buf;
          checkCudaErrors(cudaMalloc((void**)&d_batch_buf, nDim * batch_size_ * sizeof(cuDoubleComplex)));
          d_batch.push_back(d_batch_buf);
        }
        
    };

    ~QBatchSimulator() {
      for (size_t i = 0; i < h_batch.size(); i++)
      {
        checkCudaErrors(cudaFreeHost(h_batch[i]));
      }
      for (int i = 0; i < d_batch.size(); i++) {
        checkCudaErrors(cudaFree(d_batch[i]));
      }
      for (int i = 0; i < fused_gates_val_d.size(); i++) {
        checkCudaErrors(cudaFree(fused_gates_val_d[i]));
        checkCudaErrors(cudaFree(fused_gates_indices_d[i]));
      }
      // for (int i = 0; i < fused_gates_val_mored.size(); i++) {
      //   checkCudaErrors(cudaFree(fused_gates_val_mored[i]));
      //   checkCudaErrors(cudaFree(fused_gates_indices_mored[i]));
      // }
    }

    void simulate() {
        bool hasNonmeasurementNonUnitary = false;
        bool hasMeasurements             = false;
        bool measurementsLast            = true;


        for (auto& op: *qc) {
            if (op->isClassicControlledOperation() || (op->isNonUnitaryOperation() && op->getType() != qc::Measure && op->getType() != qc::Barrier)) {
                hasNonmeasurementNonUnitary = true;
            }
            if (op->getType() == qc::Measure) {
                auto* nonUnitaryOp = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                if (nonUnitaryOp == nullptr) {
                    throw std::runtime_error("Op with type Measurement could not be casted to NonUnitaryOperation");
                }
                hasMeasurements = true;

                const auto& quantum = nonUnitaryOp->getTargets();
                const auto& classic = nonUnitaryOp->getClassics();

                if (quantum.size() != classic.size()) {
                    throw std::runtime_error("Measurement: Sizes of quantum and classic register mismatch.");
                }

            }

            if (hasMeasurements && op->isUnitary()) {
                measurementsLast = false;
            }
        }

        // easiest case: all gates are unitary --> simulate once and sample away on all qubits
        if (!hasNonmeasurementNonUnitary && !hasMeasurements) {
            singleShot(false);
            return;
        }

        // single shot is enough, but the sampling should only return actually measured qubits
        if (!hasNonmeasurementNonUnitary && measurementsLast) {
            singleShot(true);
            const auto                         qubits = qc->getNqubits();
            const auto                         cbits  = qc->getNcbits();

            return;
        }
        return;
    }


    void singleShot(bool ignoreNonUnitaries) {
        std::size_t                 opNum = 0;
        std::vector<int> fused_num_nonzero;
        std::vector<qc::FusedGate> fused_gates;

        auto begin_fusion = std::chrono::high_resolution_clock::now();
        qc::CircuitOptimizer::GateFusion(std::move(qc), fused_gates, std::move(QBatchSimulator<Config>::dd), nDim, true);
        auto end_fusion = std::chrono::high_resolution_clock::now();
        std::cout << "[Stage 1: Gate Fusion] time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_fusion - begin_fusion).count() << std::endl;
        
        auto begin_convert = std::chrono::high_resolution_clock::now();
        int total_macs = 0;
        int *sparse_idx_x;
        checkCudaErrors(cudaMallocHost((void**)&sparse_idx_x, nDim* sizeof(int)));
        cuDoubleComplex *fused_gate_val_h;
        int * fused_gate_indices_h;
        for (int idx = 0; idx < fused_gates.size(); idx++) {
          qc::FusedGate fused_gate = fused_gates[idx];
          fused_num_nonzero.push_back(fused_gate.num_mac );
          total_macs += fused_gate.num_mac;

          std::cout << "Converting fused gate #" << idx << " using ";
          auto begin_gate_convert = std::chrono::high_resolution_clock::now();
          if ((ddell_conversion == DDELL_Mixed && fused_gate.num_edges < conversion_edge_thresh) || (ddell_conversion == DDELL_GPU)) {
            std::cout << "GPU" << std::endl;
            cuDoubleComplex *fused_gate_val;
            int *fused_gate_indices;
            dd::GPU_DD_edge* d_edge_arr;
            dd::GPU_DD_node* d_node_arr; 
            int num_edges = fused_gate.num_edges;
            int num_nodes = fused_gate.num_nodes;
            if (fused_gate.num_mac  > MAX_DECODED_MACS) {
              std::cerr << "[ERROR] Num of decoded MACs" << fused_gate.num_mac   << "exceeded limit!\n";
            }
            dd::GPU_DD_edge* h_edge_arr;
            dd::GPU_DD_node* h_node_arr;
            checkCudaErrors(cudaMallocHost((void**)&h_edge_arr, num_edges* sizeof(dd::GPU_DD_edge)));
            checkCudaErrors(cudaMallocHost((void**)&h_node_arr, num_nodes* sizeof(dd::GPU_DD_node)));
            // DFS GPU struct. construction
            QBatchSimulator<Config>::dd->DFS_fill_gpu_structure(fused_gate.fused_edge, h_edge_arr, h_node_arr);

            checkCudaErrors(cudaMalloc((void**)&d_edge_arr, num_edges* sizeof(dd::GPU_DD_edge)));
            checkCudaErrors(cudaMalloc((void**)&d_node_arr, num_nodes* sizeof(dd::GPU_DD_node)));
            checkCudaErrors(cudaMemcpy(d_edge_arr, h_edge_arr, num_edges* sizeof(dd::GPU_DD_edge), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_node_arr, h_node_arr, num_nodes* sizeof(dd::GPU_DD_node), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, fused_gate.num_mac  *nDim* sizeof(int)));

            // kernel
            dd_extract_matrix<<<nDim, MAX_LEV>>>(
              d_edge_arr, d_node_arr, fused_gate_val, fused_gate_indices,
              num_nodes, num_edges,  fused_gate.num_mac, qc->getNqubits()
            );
            // dd_extract_matrix_warp<<<nDim/WARPS_PER_BLOCK, WARP_SIZE*WARPS_PER_BLOCK>>>(
            //   d_edge_arr, d_node_arr, fused_gate_val, fused_gate_indices,
            //   num_nodes, num_edges,  fused_gate.num_mac, qc->getNqubits()
            // );
            checkCudaErrors( cudaDeviceSynchronize() );
            fused_gates_val_d.push_back(fused_gate_val);
            fused_gates_indices_d.push_back(fused_gate_indices);
            checkCudaErrors(cudaFreeHost(h_edge_arr));
            checkCudaErrors(cudaFreeHost(h_node_arr));
            checkCudaErrors(cudaFree(d_edge_arr));
            checkCudaErrors(cudaFree(d_node_arr));
          }
          else {
            std::cout << "CPU" << std::endl;
            checkCudaErrors(cudaMallocHost((void**)&fused_gate_val_h, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMallocHost((void**)&fused_gate_indices_h, fused_gate.num_mac  *nDim* sizeof(int)));
            memset(sparse_idx_x, 0, nDim * sizeof(int));
            QBatchSimulator<Config>::dd->dd_extract_matrix_cpu(fused_gate.fused_edge, fused_gate_val_h, 
              fused_gate_indices_h, 0, 0, sparse_idx_x, fused_gate.num_mac, {1, 0});
            
            ////////////////////////////////////////////
            // new experiment: nzr uniform distribution
            // std::unordered_map<int, int> nzr_map;
            // for (size_t row_itr = 0; row_itr < nDim; row_itr++)
            // {
            //   int nzr = 0;
            //   for (size_t col_iter = 0; col_iter < fused_gate.num_mac; col_iter++)
            //   {
            //     if (fused_gate_val_h[row_itr * fused_gate.num_mac + col_iter].x != 0 || 
            //         fused_gate_val_h[row_itr * fused_gate.num_mac + col_iter].y != 0) {
            //       nzr++;
            //     }
            //   }
            //   if (nzr_map.find(nzr) != nzr_map.end()) {
            //     nzr_map[nzr] = nzr_map[nzr]+1;
            //   }
            //   else {
            //     nzr_map.insert({nzr, 1});
            //   }
            // }
            // for (auto nzr_pair : nzr_map)
            //   std::cout <<"  NZR Pair: " << nzr_pair.first << ", " << nzr_pair.second << '\n';

            ////////////////////////////////////////////
            cuDoubleComplex *fused_gate_val;
            int *fused_gate_indices;
            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, fused_gate.num_mac  *nDim* sizeof(int)));
            checkCudaErrors(cudaMemcpy(fused_gate_val, fused_gate_val_h, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(fused_gate_indices, fused_gate_indices_h, fused_gate.num_mac * nDim* sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaFreeHost(fused_gate_val_h));
            checkCudaErrors(cudaFreeHost(fused_gate_indices_h));
            fused_gates_val_d.push_back(fused_gate_val);
            fused_gates_indices_d.push_back(fused_gate_indices);
          }
          auto end_gate_convert = std::chrono::high_resolution_clock::now();
          std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_gate_convert - begin_gate_convert).count() 
          << " " << fused_gate.num_nodes << " " << fused_gate.num_edges << " " << qc->getNqubits() <<std::endl;
        }
        checkCudaErrors(cudaFreeHost(sparse_idx_x));
        auto end_convert = std::chrono::high_resolution_clock::now();
        std::cout << "[Stage 2: DD-to-ELL Conversion] time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count() << std::endl;

        if (export_fused_gates == true)
        {
          std::ofstream outputFile("../../log/fused_gates/"+qc->getName()+".txt");
          std::vector<std::vector<int>> tgt_qubits_gates;
          std::vector<std::vector<dd::ComplexValue>> tensor_gates;
          for (int idx = 0; idx < fused_gates.size(); idx++) {
            printf("exporting fused gate %d to tensor\n", idx);
            qc::FusedGate fused_gate = fused_gates[idx];
            QBatchSimulator<Config>::dd->DD_matrix_extract(fused_gate.fused_edge, tgt_qubits_gates, tensor_gates);
          }
          outputFile << fused_gates.size() << "\n";
          for (int idx = 0; idx < fused_gates.size(); idx++) {
            printf("exporting fused gate tensor %d to file\n", idx);
            outputFile << tgt_qubits_gates[idx].size() << "\n";
            outputFile << tgt_qubits_gates[idx][0];
            for (int tgt_idx = 1; tgt_idx < tgt_qubits_gates[idx].size(); tgt_idx++) {
              outputFile << " " << tgt_qubits_gates[idx][tgt_idx];
            }
            outputFile << "\n";
            outputFile << tensor_gates[idx].size() << "\n";
            outputFile << tensor_gates[idx][0].r << " " << tensor_gates[idx][0].i;
            for (int ten_idx = 1; ten_idx < tensor_gates[idx].size(); ten_idx++) {
              outputFile << " " << tensor_gates[idx][ten_idx].r << " " << tensor_gates[idx][ten_idx].i;
            }
            outputFile << "\n";
          }
          outputFile.close();
        }
        
        ///////////////////////////////////////////
        printf("fused gates num. = %d\n", fused_num_nonzero.size());
        printf("total macs = %d\n", total_macs);


        auto begin_sim = std::chrono::high_resolution_clock::now();
        tf::Taskflow taskflow("ELL-sim");
        tf::Executor executor;

        taskflow.emplace([&](){
          tf::cudaFlow cudaflow;
          std::vector<tf::cudaTask> input_copies;
          std::vector<tf::cudaTask> output_copies;
          std::vector<tf::cudaTask> simulate_fused_gate;
          std::vector<tf::cudaTask> gate_val_copies;
          std::vector<tf::cudaTask> gate_indices_copies;
          input_copies.reserve(num_batch);
          output_copies.reserve(num_batch);
          simulate_fused_gate.reserve(num_batch*fused_num_nonzero.size());
          // int grid_size = (nDim / (MAX_CUDA_THREADS_PER_BLOCK/batch_size) > 8192)?8192:(nDim/(MAX_CUDA_THREADS_PER_BLOCK/batch_size));
          // dim3 block_size = dim3(batch_size, MAX_CUDA_THREADS_PER_BLOCK/batch_size, 1);
          int grid_size = (nDim > 8192)?8192:nDim;
          dim3 block_size = dim3(batch_size, 1, 1);

          // Fill the graph nodes
          for (int batch_id = 0; batch_id < num_batch; batch_id++) {
            // input_copies.emplace_back(cudaflow.copy(
            //   d_batch[(batch_id%2)*2+(batch_id*(fused_num_nonzero.size()+1))%2], h_batch[0], nDim * batch_size
            // ).name("input_H2D_Host->"+std::to_string((batch_id*(fused_num_nonzero.size()+1))%2)));
            input_copies.emplace_back(cudaflow.copy(
              d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1))%2], h_batch[0], nDim * batch_size
            ).name("input_H2D_Host->"+std::to_string((batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1))%2)));

            for (opNum = 0; opNum < fused_num_nonzero.size(); opNum++) {
              // simulate_fused_gate.emplace_back(cudaflow.kernel(
              //   grid_size,
              //   block_size,
              //   0,
              //   run_fused_gate,
              //   fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
              //   d_batch[(batch_id%2)*2+(batch_id*(fused_num_nonzero.size()+1)+opNum)%2], d_batch[(batch_id%2)*2+(batch_id*(fused_num_nonzero.size()+1)+opNum+1)%2], batch_size, nDim
              // ).name("fused_gate_"+std::to_string(opNum)));
              simulate_fused_gate.emplace_back(cudaflow.kernel(
                grid_size,
                block_size,
                0,
                run_fused_gate,
                fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
                d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+opNum)%2], 
                d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+opNum+1)%2], batch_size, nDim
              ).name("fused_gate_"+std::to_string(opNum)));
            }

            // output_copies.emplace_back(cudaflow.copy(
            //   h_batch[1], d_batch[(batch_id%2)*2+((batch_id+1)*fused_num_nonzero.size()+batch_id)%2], nDim * batch_size
            // ).name("output_D2H_"+std::to_string(((batch_id+1)*fused_num_nonzero.size()+batch_id)%2)+"->Host"));
            output_copies.emplace_back(cudaflow.copy(
              h_batch[1], d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+fused_num_nonzero.size())%2], nDim * batch_size
            ).name("output_D2H_"+std::to_string((batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+fused_num_nonzero.size())%2)+"->Host"));
          }

          // Dependencies
          for (int batch_id = 0; batch_id < num_batch; batch_id++) {
            // dependencies between H2D and the kernels
            input_copies[batch_id].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
            if (batch_id > 1) {
              simulate_fused_gate[(batch_id-1)*fused_num_nonzero.size()-1].precede(input_copies[batch_id]);
            }

            // dependencies within the kernels
            if (batch_id > 0) {
              simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
            }
            for (opNum = 1; opNum < fused_num_nonzero.size(); opNum++) {
              simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
            }

            // dependencies between D2H and the kernels
            simulate_fused_gate[(batch_id+1)*fused_num_nonzero.size()-1].precede(output_copies[batch_id]);
            if (batch_id < num_batch-2) {
              output_copies[batch_id].precede(simulate_fused_gate[(batch_id+2)*fused_num_nonzero.size()]);
            }
          }
          
          // else {
          //   // Fill the graph nodes
          //   for (int batch_id = 0; batch_id < num_batch; batch_id++) {
          //     input_copies.emplace_back(cudaflow.copy(
          //       d_batch[(batch_id%2)*2+0], h_batch[0], nDim * batch_size
          //     ).name("input_H2D_0->"+std::to_string((batch_id%2)*2+0)));
          //     for (opNum = 0; opNum < gpu_full_at; opNum++) {
          //       simulate_fused_gate.emplace_back(cudaflow.kernel(
          //         grid_size,
          //         block_size,
          //         0,
          //         run_fused_gate,
          //         fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
          //         d_batch[(batch_id%2)*2+opNum%2], d_batch[(batch_id%2)*2+(opNum+1)%2], batch_size, nDim
          //       ).name("fused_gate_"+std::to_string(opNum)));
          //     }
          //     for (opNum = gpu_full_at; opNum < fused_num_nonzero.size(); opNum++) {
          //       gate_val_copies.push_back(cudaflow.copy(
          //         fused_gates_val_mored[(opNum-gpu_full_at)%2], fused_gates_val_moreh[(opNum-gpu_full_at)], nDim * fused_num_nonzero[opNum]
          //       ).name("gate_val_H2D"));
          //       gate_indices_copies.push_back(cudaflow.copy(
          //         fused_gates_indices_mored[(opNum-gpu_full_at)%2], fused_gates_indices_moreh[(opNum-gpu_full_at)], nDim * fused_num_nonzero[opNum]
          //       ).name("gate_indices_H2D"));
          //       simulate_fused_gate.emplace_back(cudaflow.kernel(
          //         grid_size,
          //         block_size,
          //         0,
          //         run_fused_gate,
          //         fused_gates_val_mored[(opNum-gpu_full_at)%2], fused_gates_indices_mored[(opNum-gpu_full_at)%2], fused_num_nonzero[opNum],
          //         d_batch[(batch_id%2)*2+opNum%2], d_batch[(batch_id%2)*2+(opNum+1)%2], batch_size, nDim
          //       ).name("fused_gate_"+std::to_string(opNum)));
          //     }

          //     output_copies.emplace_back(cudaflow.copy(
          //       h_batch[1], d_batch[(batch_id%2)*2+opNum%2], nDim * batch_size
          //     ).name("output_D2H_"+std::to_string((batch_id%2)*2+opNum%2)+"->1"));
          //   }

          //   // Dependencies
          //   for (int batch_id = 0; batch_id < num_batch; batch_id++) {
          //     // dependencies between H2D and the kernels
          //     input_copies[batch_id].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
          //     if (batch_id > 1) {
          //       simulate_fused_gate[(batch_id-1)*fused_num_nonzero.size()-1].precede(input_copies[batch_id]);
          //     }

          //     // // dependencies within the kernels
          //     if (batch_id > 0) {
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-2].precede(gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+1]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-2].precede(gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+1]);
          //     }

          //     for (opNum = 1; opNum < gpu_full_at; opNum++) {
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //     }
              
          //     for (opNum = gpu_full_at; opNum < fused_num_nonzero.size(); opNum++) {
          //       gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //       gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //       if (opNum - gpu_full_at > 1) {
          //         simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-2].precede(gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at]);
          //         simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-2].precede(gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at]);
          //       }
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //     }

          //     // dependencies between D2H and the kernels
          //     simulate_fused_gate[(batch_id+1)*fused_num_nonzero.size()-1].precede(output_copies[batch_id]);
          //     if (batch_id < num_batch-2) {
          //       output_copies[batch_id].precede(simulate_fused_gate[(batch_id+2)*fused_num_nonzero.size()]);
          //     }
          //   }
          // }
 
          tf::cudaStream stream;
          cudaflow.run(stream);
          stream.synchronize(); 
          // cudaflow.dump(std::cout); 
        });

        executor.run(taskflow).wait();

        QBatchSimulator<Config>::final_state_idx = 1;
        QBatchSimulator<Config>::final_state_idx_gpu = ((num_batch-1)%2)*2+(((num_batch-1)/2)*(fused_num_nonzero.size()+1)+fused_num_nonzero.size())%2;
        auto end_sim = std::chrono::high_resolution_clock::now();
        std::cout << "[Stage 3: ELL-based batch simulation] time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_sim - begin_sim).count() << std::endl;
    }

    [[nodiscard]]
    cuDoubleComplex* getVector() const {
        if (getNumberOfQubits() >= MAX_LEV) {
            // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
            throw std::range_error("getVector only supports less than 60 qubits.");
        }
        return h_batch[final_state_idx];
    }

    [[nodiscard]] std::size_t getNumberOfQubits() const { return qc->getNqubits(); };

    [[nodiscard]] std::size_t getNumberOfOps() const { return qc->getNops(); };

    [[nodiscard]] std::string getName() const { return qc->getName(); };

    std::unique_ptr<dd::Package<Config>>     dd  = std::make_unique<dd::Package<Config>>();
    std::vector<cuDoubleComplex *> h_batch, d_batch;

    int                                     final_state_idx;
    int        final_state_idx_gpu;

    unsigned int                            fuse = 0;
    size_t nDim = 1;
    bool export_fused_gates = false;
    int ddell_conversion = 2;
    int conversion_edge_thresh = 2000;

protected:
    dd::fp        epsilon = 0.001;
    std::unique_ptr<qc::QuantumComputation> qc;
    std::size_t                             singleShots{0};
    int batch_size = 1;
    int num_batch = 1;
    int gpu_full_at = -1;
    std::vector<cuDoubleComplex*> fused_gates_val_d;
    std::vector<int*> fused_gates_indices_d;


    // std::vector<cuDoubleComplex*> fused_gates_val_moreh;
    // std::vector<int*> fused_gates_indices_moreh;
    // std::vector<cuDoubleComplex*> fused_gates_val_mored;
    // std::vector<int*> fused_gates_indices_mored;

    // 
};

template class QBatchSimulator<dd::DDPackageConfig>;

#endif //QBATCH_SIMULATOR_H
