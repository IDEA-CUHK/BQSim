#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "operations/Operation.hpp"
#include "dd/Package.hpp"
#include "dd/Export.hpp"
#include "dd/Operations.hpp"

#include <array>
#include <memory>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define MAX_CUDA_THREADS_PER_BLOCK 1024
#define MAX_DECODED_MACS 50 // for every thread
#define MAX_VAL 1024
#define MAX_LEV 40
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 2
#define TWO_GB 2
#define THRESH_GB 49526865921

__global__ void ELL_max_row(
  dd::GPU_DD_edge* dd_edges,
  dd::GPU_DD_node* dd_nodes,
  int *ell_rows,
  int num_nodes,
  int num_edges,
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
    ell_rows[bid] = decode_ptr[0];
  }
}

__global__ void Max_Sequential_Addressing_Shared(int* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                int lhs = sdata[threadIdx.x];
                int rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}

namespace qc {
static constexpr std::array<qc::OpType, 10> DIAGONAL_GATES = {
    qc::Barrier, qc::I,    qc::Z,     qc::S,  qc::Sdag,
    qc::T,       qc::Tdag, qc::Phase, qc::RZ, qc::RZZ};

static constexpr std::array<qc::OpType, 12> PERM_GATES = {
    qc::Barrier, qc::I,    qc::Z,     qc::S,  qc::Sdag,
    qc::T,       qc::Tdag, qc::Phase, qc::RZ, qc::RZZ, 
    qc::X, qc::Y};

class FusedGate {
public:
  dd::mEdge fused_edge;
  int num_edges;
  int num_nodes;
  int num_mac;
  bool perm_or_dense;

  FusedGate() = default;
  FusedGate(dd::mEdge _fused_edge, 
  int _num_mac,
  bool _perm_or_dense, std::unique_ptr<dd::Package<dd::DDPackageConfig>>&&    dd) :
  fused_edge(_fused_edge), 
  num_mac(_num_mac), perm_or_dense(_perm_or_dense)
  { 
    num_nodes = dd->node_count(_fused_edge);
    num_edges = dd->edge_count(_fused_edge);
  }
};

class CircuitOptimizer {
// protected:
//   static void addToDag(DAG& dag, std::unique_ptr<Operation>* op);
//   static void addNonStandardOperationToDag(DAG& dag,
//                                            std::unique_ptr<Operation>* op);

public:
  CircuitOptimizer() = default;

  // static DAG constructDAG(QuantumComputation& qc);
  // static void printDAG(const DAG& dag);
  // static void printDAG(const DAG& dag, const DAGIterators& iterators);

  // static void swapReconstruction(QuantumComputation& qc);

  // static void singleQubitGateFusion(QuantumComputation& qc);

  static void GateFusion(std::unique_ptr<QuantumComputation> &&qc, std::vector<FusedGate>& fused3,
 std::unique_ptr<dd::Package<dd::DDPackageConfig>>&&    dd, size_t nDim, bool fuse) {
  if (fuse) {
  // stage 1: fuse the perms
    std::vector<FusedGate> fused1;
    std::vector<dd::mEdge> perm_holder;
    bool on_perm = false;

    for (int cur_op = 0; cur_op < qc->ops.size() && !(qc->ops[cur_op]->isNonUnitaryOperation()); cur_op++) { 
      auto cur_dd = dd::getDD(qc->ops[cur_op].get(), dd);
      // if not a perm gate
      if (std::find(PERM_GATES.begin(), PERM_GATES.end(), qc->ops[cur_op]->getType()) ==
        PERM_GATES.end())
      {
        if (on_perm) {
          on_perm = false;
          // collect the fused perm first

          fused1.emplace_back(FusedGate(perm_holder[0],  1, true, std::move(dd)));
          perm_holder.clear();
        }
        // collect this gate
        std::unordered_map<dd::mNode*, dd::vEdge> mac_map; 
        size_t num_mac = dd->MaxELLRowDD(cur_dd, mac_map);
        fused1.emplace_back(FusedGate(cur_dd, num_mac, false, std::move(dd)));
        mac_map.clear();
      }
      else {
        on_perm = true;
        
        if (perm_holder.size() == 0)
          perm_holder.emplace_back(cur_dd);
        else {
          auto cur_fused_perm = dd->multiply(cur_dd, perm_holder[0]);
          perm_holder[0] = cur_fused_perm;
        }
      }
    }
    if (on_perm) {
      fused1.emplace_back(FusedGate(perm_holder[0], 1, true, std::move(dd)));
      perm_holder.clear();
    }

    // stage 2: fuse two consecutive dense gates on two qubits
    // std::vector<bool> perm_gate2; // true: perm, false: dense
    std::vector<FusedGate> fused2;
    std::vector<dd::mEdge> dense_holder;
    for (int cur_op = 0; cur_op < fused1.size(); cur_op++) {
      if (fused1[cur_op].perm_or_dense == false) {
        if (dense_holder.size() >0) {
          auto cur_fused_perm = dd->multiply(fused1[cur_op].fused_edge, dense_holder[0]);
          std::unordered_map<dd::mNode*, dd::vEdge> mac_map; 
          size_t num_mac = dd->MaxELLRowDD(cur_fused_perm, mac_map);
          fused2.emplace_back(FusedGate(cur_fused_perm, num_mac, false, std::move(dd)));
          mac_map.clear();
          dense_holder.clear();
        }
        else {
          dense_holder.emplace_back(fused1[cur_op].fused_edge);
        }
      }
      else {
        if (dense_holder.size() >0) {
          std::unordered_map<dd::mNode*, dd::vEdge> mac_map; 
          size_t num_mac = dd->MaxELLRowDD(dense_holder[0], mac_map);
          fused2.emplace_back(FusedGate(dense_holder[0], num_mac, false, std::move(dd)));
          mac_map.clear();
          dense_holder.clear();
        }
        fused2.emplace_back(fused1[cur_op]);
      }
    }
    if (dense_holder.size() >0) {
      std::unordered_map<dd::mNode*, dd::vEdge> mac_map; 
      size_t num_mac = dd->MaxELLRowDD(dense_holder[0], mac_map);
      fused2.emplace_back(FusedGate(dense_holder[0], num_mac, false, std::move(dd)));
      mac_map.clear();
      dense_holder.clear();
    }
    fused1.clear();

    // stage 3: fuse the rest (start from) (try greedy first)
    bool start_fuse = false;
    int prev_mac = fused2[0].num_mac;
    auto prev_dd = fused2[0].fused_edge;
    std::unordered_map<dd::mNode*, dd::vEdge> mac_map; 
    for (int idx = 1; idx < fused2.size(); idx++) {
      std::cout << "fusing " << idx << " out of " << fused2.size() << std::endl;
      // if (!start_fuse && !(fused2[idx].perm_or_dense)) {
      //   fused3.emplace_back(fused2[idx]);
      //   prev_mac = fused2[idx].num_mac;
      //   prev_dd = fused2[idx].fused_edge;
      // }
      // else {
        // start_fuse = true;
        size_t cur_mac = fused2[idx].num_mac;
        auto cur_fused = dd->multiply(fused2[idx].fused_edge, prev_dd);
        // int fused_mac = MaxELLRowGPU(cur_fused, std::move(dd), nDim, qc->getNqubits());
        size_t fused_mac = dd->MaxELLRowDD(cur_fused, mac_map);
        if (prev_mac + cur_mac < fused_mac) {
          fused3.emplace_back(FusedGate(prev_dd, prev_mac, false, std::move(dd)));
          prev_mac = cur_mac;
          prev_dd = fused2[idx].fused_edge;
        }
        else {
          prev_mac = fused_mac;
          prev_dd = cur_fused;
        }
      // }
    }
    mac_map.clear();
    fused3.emplace_back(FusedGate(prev_dd, prev_mac, false, std::move(dd)));
    // // TODO: stage 3: try DP with pruned search space

    fused2.clear();
    std::cout << "Average Coefficient of variation: " << dd->cv_accum/ dd->fuse_cnt << std::endl; 
  }
  else {
    for (int cur_op = 0; cur_op < qc->ops.size() && !(qc->ops[cur_op]->isNonUnitaryOperation()); cur_op++) { 
      auto cur_dd = dd::getDD(qc->ops[cur_op].get(), dd);
      std::unordered_map<dd::mNode*, dd::vEdge> mac_map; 
      size_t num_mac = dd->MaxELLRowDD(cur_dd, mac_map);
      mac_map.clear();
      fused3.emplace_back(FusedGate(cur_dd, num_mac, false, std::move(dd)));
    }
  }
 
}

static int MaxELLRowGPU(dd::mEdge fused_edge,
 std::unique_ptr<dd::Package<dd::DDPackageConfig>> && dd,
 size_t nDim, int num_qubits)  {
  int *ell_rows;
  int max_ell_row[1];
  checkCudaErrors(cudaMalloc((void**)&ell_rows, nDim* sizeof(int)));
  dd::GPU_DD_edge* d_edge_arr;
  dd::GPU_DD_node* d_node_arr;
  int num_nodes = dd->node_count(fused_edge);
  int num_edges = dd->edge_count(fused_edge);
  dd::GPU_DD_edge* h_edge_arr;
  dd::GPU_DD_node* h_node_arr;
  checkCudaErrors(cudaMallocHost((void**)&h_edge_arr, num_edges* sizeof(dd::GPU_DD_edge)));
  checkCudaErrors(cudaMallocHost((void**)&h_node_arr, num_nodes* sizeof(dd::GPU_DD_node)));

  dd->DFS_fill_gpu_structure(fused_edge, h_edge_arr, h_node_arr);

  checkCudaErrors(cudaMalloc((void**)&d_edge_arr, num_edges* sizeof(dd::GPU_DD_edge)));
  checkCudaErrors(cudaMalloc((void**)&d_node_arr, num_nodes* sizeof(dd::GPU_DD_node)));
  checkCudaErrors(cudaMemcpy(d_edge_arr, h_edge_arr, num_edges* sizeof(dd::GPU_DD_edge), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_node_arr, h_node_arr, num_nodes* sizeof(dd::GPU_DD_node), cudaMemcpyHostToDevice));

  ELL_max_row<<<nDim, MAX_LEV>>>(d_edge_arr, d_node_arr, ell_rows,
                num_nodes, num_edges, num_qubits);
  checkCudaErrors( cudaDeviceSynchronize() );
  Max_Sequential_Addressing_Shared<<<nDim / MAX_CUDA_THREADS_PER_BLOCK, MAX_CUDA_THREADS_PER_BLOCK>>>(ell_rows, nDim);
  checkCudaErrors( cudaDeviceSynchronize() );
  checkCudaErrors(cudaMemcpy(max_ell_row, ell_rows, sizeof(int), cudaMemcpyDeviceToHost));
  int mac_num = max_ell_row[0];

  checkCudaErrors(cudaFreeHost(h_edge_arr));
  checkCudaErrors(cudaFreeHost(h_node_arr));
  checkCudaErrors(cudaFree(d_edge_arr));
  checkCudaErrors(cudaFree(d_node_arr));
  checkCudaErrors(cudaFree(ell_rows));

  return mac_num;
}

//   static void removeIdentities(QuantumComputation& qc);

//   static void removeDiagonalGatesBeforeMeasure(QuantumComputation& qc);

//   static void removeFinalMeasurements(QuantumComputation& qc);

//   static void decomposeSWAP(QuantumComputation& qc,
//                             bool isDirectedArchitecture);

//   static void decomposeTeleport(QuantumComputation& qc);

//   static void eliminateResets(QuantumComputation& qc);

//   static void deferMeasurements(QuantumComputation& qc);

//   static bool isDynamicCircuit(QuantumComputation& qc);

//   static void reorderOperations(QuantumComputation& qc);

//   static void flattenOperations(QuantumComputation& qc);

//   static void cancelCNOTs(QuantumComputation& qc);

// protected:
//   static void removeDiagonalGatesBeforeMeasureRecursive(
//       DAG& dag, DAGReverseIterators& dagIterators, Qubit idx,
//       const qc::Operation* until);
//   static bool removeDiagonalGate(DAG& dag, DAGReverseIterators& dagIterators,
//                                  Qubit idx, DAGReverseIterator& it,
//                                  qc::Operation* op);

//   static void
//   removeFinalMeasurementsRecursive(DAG& dag, DAGReverseIterators& dagIterators,
//                                    Qubit idx, const qc::Operation* until);
//   static bool removeFinalMeasurement(DAG& dag,
//                                      DAGReverseIterators& dagIterators,
//                                      Qubit idx, DAGReverseIterator& it,
//                                      qc::Operation* op);

//   static void changeTargets(Targets& targets,
//                             const std::map<Qubit, Qubit>& replacementMap);
//   static void changeControls(Controls& controls,
//                              const std::map<Qubit, Qubit>& replacementMap);

//   using Iterator = decltype(qc::QuantumComputation::ops.begin());
//   static Iterator
//   flattenCompoundOperation(std::vector<std::unique_ptr<Operation>>& ops,
//                            Iterator it);
};
} // namespace qc
