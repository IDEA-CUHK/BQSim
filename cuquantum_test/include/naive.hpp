// #pragma once

// #include "base.hpp"
// #include <chrono>

// // #include <util/parser.hpp>
// using namespace std::chrono;

// __global__ void naive_two_qubit_gate(
//     cuDoubleComplex gate_11,
//     cuDoubleComplex gate_12,
//     cuDoubleComplex gate_21,
//     cuDoubleComplex gate_22,
//     int batchSize,
//     int nqubits,
//     int ctrl,
//     int target,
//     cuDoubleComplex *input_state,
//     cuDoubleComplex *output_state
// )
// {
//   uint64_t mask = 1ULL << target;
//   uint64_t lo_mask = mask - 1;
//   uint64_t hi_mask = ~lo_mask;
//   uint tid = threadIdx.x;
//   uint bid = blockIdx.x;
//   uint i0 = ((bid & hi_mask) << 1) | (bid & lo_mask);
//   uint i1 = i0 | mask;
  
//   cuDoubleComplex src1 = input_state[tid + i0 * batchSize];
//   cuDoubleComplex src2 = input_state[tid + i1 * batchSize];
//   if (ctrl != -1) {
//     uint apply_flag = i0 & (1ULL << ctrl);

//     if (apply_flag) {
//       cuDoubleComplex dst1 = cuCadd(cuCmul(gate_11, src1), cuCmul(gate_12, src2));
//       cuDoubleComplex dst2 = cuCadd(cuCmul(gate_21, src1), cuCmul(gate_22, src2));

//       output_state[tid + i0 * batchSize] = dst1;
//       output_state[tid + i1 * batchSize] = dst2;
//     }
//     else {
//       output_state[tid + i0 * batchSize] = src1;
//       output_state[tid + i1 * batchSize] = src2;
//     }
//   }
//   else {
//     cuDoubleComplex dst1 = cuCadd(cuCmul(gate_11, src1), cuCmul(gate_12, src2));
//     cuDoubleComplex dst2 = cuCadd(cuCmul(gate_21, src1), cuCmul(gate_22, src2));
//     output_state[tid + i0 * batchSize] = dst1;
//     output_state[tid + i1 * batchSize] = dst2;
//   }
// }

// class NaiveQBatch : public Base
// {

// public:
//   NaiveQBatch(    
//     int _n_qubit,
//     int _nSvSize,
//     int _batchSize,
//     std::vector<qpp::QCircuit::double2 *> _mat_vec,
//     std::vector<int> _ctrl_vec,
//     std::vector<int> _target_vec,
//     int _n_batch
//   );
//   void BatchSim() override;
//   void ReadInputs() override;
//   ~NaiveQBatch();
// };

// void NaiveQBatch::ReadInputs() {
//   std::ifstream file;
//   file.open((filename).c_str());

//   if (!file.is_open()) {
//     std::cerr << "Failed to open file." << std::endl;
//     exit(-1);
//   }
//   std::string line;
//   int batch_id = 0;
//   while (getline(file, line)) {
//     std::istringstream iss(line);
//     double real, imag;
//     int amp_id = 0;
//     while (iss >> real >> imag) {
//       input_arr[batch_id+amp_id*batchSize] = {real, imag};
//       amp_id++;
//     }
//     batch_id++;
//   }

//   file.close();

//   // for (int sv_id = 0; sv_id < batchSize; sv_id++) {
//   //   std::cout<<"[Naive] validating state vector #"<<sv_id << std::endl;
//   //   double svsum = 0;
//   //   for (int amp_id = 0; amp_id < nSvSize; amp_id++) {
//   //     svsum += (input_arr[sv_id+amp_id*batchSize].x*input_arr[sv_id+amp_id*batchSize].x
//   //       + input_arr[sv_id+amp_id*batchSize].y*input_arr[sv_id+amp_id*batchSize].y);
//   //   }
//   //   std::cout << "  state vector sum: "<< svsum << std::endl;
//   // }
 
// }

// NaiveQBatch::NaiveQBatch(    
//   int _n_qubit,
//   int _nSvSize,
//   int _batchSize,
//   std::vector<qpp::QCircuit::double2 *> _mat_vec,
//   std::vector<int> _ctrl_vec,
//   std::vector<int> _target_vec,
//   int _n_batch
// ) : Base(_n_qubit, _nSvSize, _batchSize, _mat_vec, _ctrl_vec, _target_vec, _n_batch)
// { }

// void NaiveQBatch::BatchSim() {
//   ReadInputs(); 
//   cuDoubleComplex *h_batchsv_result;
//   cudaMallocHost((void**)&h_batchsv_result, nSvSize * batchSize * n_batch * sizeof(cuDoubleComplex));
// //  memset(h_batchsv, 0, sizeof(cuDoubleComplex)*batchSize*nSvSize);
//  // h_batchsv[0] = {1.0, 0.0};
//   std::vector<cuDoubleComplex *> d_batchsv;
//   cuDoubleComplex * d_batchsv_ptr0;
//   cuDoubleComplex * d_batchsv_ptr1;

//   cudaMalloc((void**)&d_batchsv_ptr0, nSvSize * batchSize * sizeof(cuDoubleComplex));
//   cudaMalloc((void**)&d_batchsv_ptr1, nSvSize * batchSize * sizeof(cuDoubleComplex));
//   d_batchsv.push_back(d_batchsv_ptr0);
//   d_batchsv.push_back(d_batchsv_ptr1);
//   custatevecHandle_t handle;
//   custatevecCreate(&handle);
//   // cudaMalloc((void**)&d_batchsv, nSvSize * batchSize * sizeof(cuDoubleComplex));
//   auto begin = std::chrono::steady_clock::now();
//   for (size_t bid = 0; bid < n_batch; bid++)
//   {
//     cudaMemcpy(d_batchsv[0], input_arr, nSvSize * batchSize * sizeof(cuDoubleComplex),
//                 cudaMemcpyHostToDevice);

//     for (int gid = 0; gid < matrix_vec.size(); gid++)  { // mat_vec.size()

//       naive_two_qubit_gate<<<nSvSize / 2, batchSize>>>(
//         matrix_vec[gid][0],
//         matrix_vec[gid][1],
//         matrix_vec[gid][2],
//         matrix_vec[gid][3],
//         batchSize,
//         n_qubit,
//         ctrl_vec[gid],
//         target_vec[gid],
//         d_batchsv[gid%2], 
//         d_batchsv[(gid+1)%2]
//       );

//     }
//     cudaMemcpy(h_batchsv_result+bid * nSvSize * batchSize, d_batchsv[matrix_vec.size()%2], nSvSize * batchSize * sizeof(cuDoubleComplex),
//                 cudaMemcpyDeviceToHost);
//   }
//   auto end = std::chrono::steady_clock::now();
//   std::cout << "Naive runtime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
//   // for (size_t i = 0; i < nSvSize * batchSize * n_batch; i++)
//   // {
//   //   std::cout << "(" << h_batchsv_result[i].x <<","<< h_batchsv_result[i].y << ") ";
//   // }
 
//   output_arr = h_batchsv_result; 
//   cudaFree(d_batchsv[0]);
//   cudaFree(d_batchsv[1]);
//   // destroy handle
//   custatevecDestroy(handle);

//   cudaFreeHost(input_arr);
//   std::cout << std::endl;
// }

// NaiveQBatch::~NaiveQBatch()
// {
//   for (size_t i = 0; i < matrix_vec.size(); i++)
//   {
//     cudaFreeHost(matrix_vec[i]);
//   }
//   cudaFreeHost(input_arr);
// }
