#include "cu_qbatch.hpp"
#include "naive.hpp"
#include <string>
#include <algorithm>
#include "util.hpp"
#include <cstddef>
#include <fstream>

void extract_fused_gate(
  std::vector<cuDoubleComplex *> &mat_vec,
  std::vector<int> &ctrl_vec,
  std::vector<std::vector<int>> &target_vec,
  std::string fused_gate_path
) {
  std::ifstream inFile(fused_gate_path);
  if (!inFile) {
      std::cerr << "Error opening file!" << std::endl;
      exit(-1);
  }
  int gate_num;
  inFile >> gate_num;
  for (int i = 0; i < gate_num; i++)
  {
    ctrl_vec.push_back(-1);
    int tgt_num;
    
    inFile >> tgt_num;
    std::vector<int> target_vec_gate;
    for (int j = 0; j < tgt_num; j++) {
      int tmp_tgt;
      inFile >> tmp_tgt;
      target_vec_gate.push_back(tmp_tgt);
    }
    target_vec.push_back(target_vec_gate);
    int mat_dim;
    inFile >> mat_dim;
    cuDoubleComplex * mat= new cuDoubleComplex[mat_dim];
    for (int j = 0; j < mat_dim; j++) {
      double tmp_mat_r, tmp_mat_i;
      inFile >> tmp_mat_r >> tmp_mat_i;
      mat[j] = {tmp_mat_r, tmp_mat_i};
    }
    mat_vec.push_back(mat);
  }
  
}

int main(int argc, char** argv) {
  using namespace qpp;
  std::string circ_name;
  int n_qubit, batchSize, n_batch, fused_gate, output_file;
  if (argc > 6) {
    circ_name = std::string(argv[1]);
    n_qubit = std::stoi(argv[2]);
    batchSize = std::stoi(argv[3]);
    n_batch = std::stoi(argv[4]);
    fused_gate = std::stoi(argv[5]);
    output_file = std::stoi(argv[6]);
  }
  else {
    std::cout<<"ERROR: arg input the file name"<<std::endl;
    std::cout << "Args: num_qubits, batchsize, num_batches, use_fused_gates: (0-no fusion, 1-qiskit fusion, 2-BQCS-aware fusion), and output_or_not (0 or 1)" << std::endl;
    return 1;
  }
  // read the circuit from the input stream



  // int n_qubit;
  if (!fused_gate) {
    std::vector<qpp::QCircuit::double2 *> mat_vec;
    std::vector<int> ctrl_vec;
    std::vector<std::vector<int>> target_vec;
    std::vector<int> _target_vec;
    QCircuit qc = qasm::read_from_file("../../circuits/"+circ_name+"_n"+std::to_string(n_qubit)+".qasm");
    qc.extract_info(mat_vec, ctrl_vec, _target_vec, n_qubit);
    for (int i = 0; i < _target_vec.size(); i++)
    {
      target_vec.push_back({_target_vec[i]});
    }
    int nSvSize    = (1 << n_qubit);
    // int batchSize  = 1;

    std::cout << "cuQuantum Baseline: "<<std::endl;
    CuQBatch cuqbatch(n_qubit, nSvSize, batchSize, mat_vec, ctrl_vec, target_vec, n_batch);
    cuqbatch.BatchSim();
    cuDoubleComplex* cuqbatch_out = cuqbatch.FetchOutput();
    std::ofstream outputFile("../../log/results/state/cuquantum"+circ_name+"_n"+std::to_string(n_qubit)+".txt");
    if (outputFile.is_open()) {
        // Write the vector to the file
        for (size_t i = 0; i < nSvSize; i++) {
            outputFile << cuqbatch_out[i].x << " " << cuqbatch_out[i].y << std::endl;
        }
        // Close the file
        outputFile.close();
        std::cout << "Data saved to file." << std::endl;
    } else {
        std::cerr << "Failed to open the file." << std::endl;
    }
    cudaFreeHost(cuqbatch_out);
  }
  else {
    std::vector<cuDoubleComplex *> mat_vec;
    std::vector<int> ctrl_vec;
    std::vector<std::vector<int>> target_vec;
    std::string fused_gate_path = "";
    if (fused_gate == 1)
      fused_gate_path = "../../log/fused_gates/qiskit_"+circ_name+"_n"+std::to_string(n_qubit)+".txt";
    else // 2
      fused_gate_path = "../../log/fused_gates/"+circ_name+"_n"+std::to_string(n_qubit)+".txt";
    extract_fused_gate(mat_vec, ctrl_vec, target_vec, fused_gate_path);
    int nSvSize    = (1 << n_qubit);
    // int batchSize  = 1;

    std::cout << "cuQuantum Baseline: "<<std::endl;
    CuQBatch cuqbatch(n_qubit, nSvSize, batchSize, mat_vec, ctrl_vec, target_vec, n_batch);
    cuqbatch.BatchSim();
    if (output_file) {
      cuDoubleComplex* cuqbatch_out = cuqbatch.FetchOutput();
      std::ofstream outputFile("../../log/results/state/cuquantum"+circ_name+"_n"+std::to_string(n_qubit)+".txt");
      if (outputFile.is_open()) {
          // Write the vector to the file
          for (size_t i = 0; i < nSvSize; i++) {
              outputFile << cuqbatch_out[i].x << " " << cuqbatch_out[i].y << std::endl;
          }
          // Close the file
          outputFile.close();
          std::cout << "Data saved to file." << std::endl;
      } else {
          std::cerr << "Failed to open the file." << std::endl;
      }
      cudaFreeHost(cuqbatch_out);
    }
  }

  return EXIT_SUCCESS;
}
