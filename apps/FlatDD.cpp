#include "SwitchSimulator.hpp"
#include "cxxopts.hpp"
#include "dd/Export.hpp"
#include "nlohmann/json.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>

namespace nl = nlohmann;

inline std::string BoolToString(bool b)
{
  return b ? "true" : "false";
}

int main(int argc, char** argv) { // NOLINT(bugprone-exception-escape)
    cxxopts::Options options("FlatDD", "FlatDD");
    // clang-format off
    options.add_options()
        ("h,help", "produce help message")
        ("pv", "save the state vector")
        ("t", "num of threads", cxxopts::value<unsigned int>()->default_value("16"))
        ("ps", "print simulation stats (applied gates, sim. time, and maximal size of the DD")
        ("file", "simulate a quantum circuit given by file (detection by the file extension)", cxxopts::value<std::string>())
        ("fuse", "enable gate fusion optimization", cxxopts::value<unsigned int>()->default_value("0"))
        ("batch_size", "number of states in a batch (integer)", cxxopts::value<int>())
        ("num_batch", "number of batches (integer)", cxxopts::value<int>())
        ("custom_inputs", "use this flag if you want to use the input from input_batch")
        ("no_cache", "no cache optimization")
        ("beta", "EMA parameter", cxxopts::value<double>()->default_value("0.9"))
        ("thresh", "set the threshold", cxxopts::value<double>()->default_value("2"))
        ("DDSIM_convert", "ddsim conversion")
        ("disable_switch", "disable the switch between DDSIM and DMAV");

    // clang-format on

    auto vm = options.parse(argc, argv);
    if (vm.count("help") > 0) {
        std::cout << options.help();
        std::exit(0);
    }

    std::unique_ptr<qc::QuantumComputation>              quantumComputation;
    std::unique_ptr<SwitchSimulator<dd::DDPackageConfig>>  swisim{nullptr};
    const bool                                           verbose = vm.count("verbose") > 0;
    
    if (vm.count("file") > 0) {
        const std::string fname = vm["file"].as<std::string>();
        quantumComputation      = std::make_unique<qc::QuantumComputation>(fname);
        swisim                   = std::make_unique<SwitchSimulator<dd::DDPackageConfig>>(std::move(quantumComputation));
    } else {
        std::cerr << "Did not find anything to simulate. See help below.\n"
                  << options.help() << "\n";
        std::exit(1);
    }

    const auto beta     = vm["beta"].as<double>();
    const auto threshold    = vm["thresh"].as<double>();
    const auto batch_size = vm["batch_size"].as<int>();
    const auto num_batch = vm["num_batch"].as<int>();

    swisim->threshold = threshold;
    const auto n_thread = vm["t"].as<unsigned int>();
    const auto fuse = vm["fuse"].as<unsigned int>();
    if (vm.count("no_cache") > 0) {
        swisim->enable_cache = false;
    }
    if (vm.count("DDSIM_convert") > 0) {
        swisim->ddsim_convert = true;
    }
    if (vm.count("custom_inputs") > 0) {
        swisim->custom_inputs = true;
    }
    if (vm.count("disable_switch") > 0) {
        swisim->enable_switch = false;
    }
    if (swisim->getNumberOfQubits() > 100) {
        std::clog << "[WARNING] Quantum computation contains quite a few qubits. You're jumping into the deep end.\n";
    }
    swisim->n_thread_exp = (unsigned int)log2(n_thread);
    swisim->fuse = fuse;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int batch_idx = 0; batch_idx < num_batch; batch_idx++)
      for (int input_idx = 0; input_idx < batch_size; input_idx++)
        swisim->simulate();
    auto t2 = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> durationSimulation = t2 - t1;
    std::cout << "Simulation finished" << std::endl;
    nl::json outputObj;

    std::vector<std::pair<dd::fp, dd::fp>> result;
    if (vm.count("pv") > 0) {
        std::vector<std::pair<dd::fp, dd::fp>> state_vector;
        dd::fp* real_vec = nullptr;
        dd::fp* imag_vec = nullptr;
        if (swisim->enable_switch) {
            if (swisim->switched)
                swisim->getVector(real_vec, imag_vec);
            else {
                swisim->getVectorFromDD(0);
                real_vec = swisim->state_real[0];
                imag_vec = swisim->state_imag[0];
            }
        }
        else {
            swisim->getVector(real_vec, imag_vec);
        }
        // outputObj["state_vector"] = state_vector; "/../../log/results/"+ddsim->getName()+"/"+ddsim->getName()+"_DDSIM
        std::ofstream outputFile("../../log/results/state/FlatDD_"+swisim->getName()+".txt");
        if (outputFile.is_open()) {
            // Write the vector to the file
            for (size_t q = 0; q < (size_t)std::pow(2, swisim->getNumberOfQubits()); q++) {
                outputFile << real_vec[q] << " " << imag_vec[q] << std::endl;
            }
            // Close the file
            outputFile.close();
            std::cout << "Data saved to file." << std::endl;
        } else {
            std::cerr << "Failed to open the file." << std::endl;
        }
    }

    outputObj["statistics"] = {
            {"simulation_time", durationSimulation.count()},
            {"benchmark", swisim->getName()},
            {"n_qubits", +swisim->getNumberOfQubits()},
            {"applied_gates", swisim->getNumberOfOps()},
            {"total_MACs", swisim->getMacs()},
            {"fused_gates", swisim->getFusedGates()},
            {"DD->Array conversion", swisim->switchTime},
            {"number of threads", n_thread}
    };

    std::vector<dd::fp> time_record_1;
    std::vector<dd::fp> time_record_2;
    dd::fp   switch_time;
    time_record_1 = swisim->getTimeRecord1();
    time_record_2 = swisim->getTimeRecord2();
    switch_time   = swisim->getSwitchTime();
    std::ofstream outputFile("../../log/results/time/"+swisim->getName()+"_FlatDD.txt");
    if (outputFile.is_open()) {
        // Write the vector to the file
        for (const auto& pair : time_record_1) {
            outputFile << pair << std::endl;
        }
        outputFile << "Switch Overhead:" << switch_time << std::endl;
        for (const auto& pair : time_record_2) {
            outputFile << pair << std::endl;
        }
        // Close the file
        outputFile.close();
        std::cout << "Time data saved to file." << std::endl;
    } else {
        std::cerr << "Failed to open the time file." << std::endl;
    }
    
    std::cout << std::setw(2) << outputObj << std::endl;
}
