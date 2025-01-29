#include "QBatchSimulator.hpp"
#include "cxxopts.hpp"
#include "dd/Export.hpp"
#include "nlohmann/json.hpp"


#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace nl = nlohmann;

int main(int argc, char** argv) { // NOLINT(bugprone-exception-escape)
    cxxopts::Options options("Quantum Batch Sim", "Quantum Batch Sim");
    // clang-format off
    options.add_options()
        ("h,help", "produce help message")
        ("pv", "save the state vector")
        ("ps", "print simulation stats (applied gates, sim. time, and maximal size of the DD")
        ("export_fused_gates", "export the fused gates")
        ("batch_size", "number of states in a batch (integer)", cxxopts::value<int>())
        ("num_batch", "number of batches (integer)", cxxopts::value<int>())
        ("conversion_type", "DD-to-ELL conversion type: GPU (0), CPU (1), Mixed (2)", cxxopts::value<int>())
        ("file", "simulate a quantum circuit given by file (detection by the file extension)", cxxopts::value<std::string>());

    // clang-format on

    auto vm = options.parse(argc, argv);
    if (vm.count("help") > 0) {
        std::cout << options.help();
        std::exit(0);
    }

    const int batch_size     = vm["batch_size"].as<int>();
    const int num_batch      = vm["num_batch"].as<int>();
    const int ddell_conversion = vm["conversion_type"].as<int>();
    // const int conversion_edge_thresh = vm["conversion_edge_thresh"].as<int>();

    std::unique_ptr<qc::QuantumComputation>              quantumComputation;
    std::unique_ptr<QBatchSimulator<dd::DDPackageConfig>>  qbatchsim{nullptr};
    const bool                                           verbose = vm.count("verbose") > 0;

    if (vm.count("file") > 0) {
        const std::string fname = vm["file"].as<std::string>();
        quantumComputation      = std::make_unique<qc::QuantumComputation>(fname);
        qbatchsim               = std::make_unique<QBatchSimulator<dd::DDPackageConfig>>(std::move(quantumComputation), batch_size, num_batch);
    } else {
        std::cerr << "Did not find anything to simulate. See help below.\n"
                  << options.help() << "\n";
        std::exit(1);
    }

    if (qbatchsim->getNumberOfQubits() > 100) {
        std::clog << "[WARNING] Quantum computation contains quite a few qubits. You're jumping into the deep end.\n";
    }
    if (vm.count("export_fused_gates") > 0) {
        qbatchsim->export_fused_gates = true;
    }
    qbatchsim->ddell_conversion = ddell_conversion;
    // qbatchsim->conversion_edge_thresh = conversion_edge_thresh;
    auto begin = std::chrono::high_resolution_clock::now();
    qbatchsim->simulate();
    auto end = std::chrono::high_resolution_clock::now();


    std::cout << "Simulation finished" << std::endl;
    // init check
    bool *identical_d;
    bool *identical_h;
    checkCudaErrors(cudaMalloc((void**)&identical_d, qbatchsim->nDim*sizeof(bool)));
    checkCudaErrors(cudaMallocHost((void**)&identical_h, qbatchsim->nDim*sizeof(bool)));
    initial_check<<<qbatchsim->nDim, batch_size, batch_size*sizeof(bool)>>>(qbatchsim->d_batch[qbatchsim->final_state_idx_gpu], identical_d, batch_size);
    checkCudaErrors(cudaMemcpy(identical_h, identical_d, qbatchsim->nDim*sizeof(bool), cudaMemcpyDeviceToHost));
    bool identical_res = true;
    for (int i = 0; i < qbatchsim->nDim; i++) {
        if (!identical_h[i]) {
        identical_res = false;
        break;
        }
    }
    std::cout << "Initial check: "<<identical_res << std::endl;
    checkCudaErrors(cudaFree(identical_d));
    checkCudaErrors(cudaFreeHost(identical_h));

    
    nl::json outputObj;

    if (vm.count("pv") > 0) {
        cuDoubleComplex* state_vector;
        state_vector = qbatchsim->getVector();
        std::ofstream outputFile("../../log/results/state/qbsim_"+qbatchsim->getName()+".txt");
        if (outputFile.is_open()) {
            // Write the vector to the file
            for (size_t i = 0; i < qbatchsim->nDim; i++) {
                outputFile << state_vector[i*batch_size].x << " " << state_vector[i*batch_size].y << std::endl;
            }
            // Close the file
            outputFile.close();
            std::cout << "Data saved to file." << std::endl;
        } else {
            std::cerr << "Failed to open the file." << std::endl;
        }
    }
    
    outputObj["statistics"] = {
            {"simulation_time", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()},
            {"benchmark", qbatchsim->getName()},
            {"n_qubits", +qbatchsim->getNumberOfQubits()},
            {"applied_gates", qbatchsim->getNumberOfOps()}
    };


    std::cout << std::setw(2) << outputObj << std::endl;
}
