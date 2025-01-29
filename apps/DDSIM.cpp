#include "DDSimulator.hpp"
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
    cxxopts::Options options("FLAT DD", "FLAT DD");
    // clang-format off
    options.add_options()
        ("h,help", "produce help message")
        ("pv", "save the state vector")
        // ("pn", "save add/mul op cnts")
        ("ps", "print simulation stats (applied gates, sim. time, and maximal size of the DD")
        // ("col", "collapse at every gate")
        ("file", "simulate a quantum circuit given by file (detection by the file extension)", cxxopts::value<std::string>());

    // clang-format on

    auto vm = options.parse(argc, argv);
    bool collapseAllGates = false;
    if (vm.count("help") > 0) {
        std::cout << options.help();
        std::exit(0);
    }
    // if (vm.count("col") > 0) {
    //     collapseAllGates = true;
    // }
    std::unique_ptr<qc::QuantumComputation>              quantumComputation;
    std::unique_ptr<DDSimulator<dd::DDPackageConfig>>  ddsim{nullptr};
    const bool                                           verbose = vm.count("verbose") > 0;

    if (vm.count("file") > 0) {
        const std::string fname = vm["file"].as<std::string>();
        quantumComputation      = std::make_unique<qc::QuantumComputation>(fname);
        ddsim                   = std::make_unique<DDSimulator<dd::DDPackageConfig>>(std::move(quantumComputation));
    } else {
        std::cerr << "Did not find anything to simulate. See help below.\n"
                  << options.help() << "\n";
        std::exit(1);
    }


    if (ddsim->getNumberOfQubits() > 100) {
        std::clog << "[WARNING] Quantum computation contains quite a few qubits. You're jumping into the deep end.\n";
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    ddsim->simulate(collapseAllGates);
    auto t2 = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> durationSimulation = t2 - t1;
    std::cout << "Simulation finished" << std::endl;
    nl::json outputObj;

    std::vector<std::pair<dd::fp, dd::fp>> result;
    if (vm.count("pv") > 0) {
        std::vector<std::pair<dd::fp, dd::fp>> state_vector;
        state_vector = ddsim->getVector<std::pair<dd::fp, dd::fp>>();
        // outputObj["state_vector"] = state_vector;
        std::ofstream outputFile("../../log/results/state/"+ddsim->getName()+"_DDSIM.txt");
        if (outputFile.is_open()) {
            // Write the vector to the file
            for (const auto& pair : state_vector) {
                outputFile << pair.first << " " << pair.second << std::endl;
            }
            // Close the file
            outputFile.close();
            std::cout << "Data saved to file." << std::endl;
        } else {
            std::cerr << "Failed to open the file." << std::endl;
        }
    }
    // if (vm.count("pn") > 0) {
    //     std::vector<size_t> mulCnt;
    //     mulCnt = ddsim->getMul();
    //     // outputObj["state_vector"] = state_vector;
    //     std::ofstream outputFile("/home/student/workspace/quantum/FLATDD/results/"+ddsim->getName()+"/"+ddsim->getName()+"_mul_ddsim.txt");
    //     if (outputFile.is_open()) {
    //         // Write the vector to the file
    //         for (const auto& em : mulCnt) {
    //             outputFile << em << std::endl;
    //         }
    //         // Close the file
    //         outputFile.close();
    //         std::cout << "Data saved to file." << std::endl;
    //     } else {
    //         std::cerr << "Failed to open the file." << std::endl;
    //     }
    //     std::vector<size_t> addCnt;
    //     addCnt = ddsim->getAdd();
    //     // outputObj["state_vector"] = state_vector;
    //     outputFile = std::ofstream("/home/student/workspace/quantum/FLATDD/results/"+ddsim->getName()+"/"+ddsim->getName()+"_add_ddsim.txt");
    //     if (outputFile.is_open()) {
    //         // Write the vector to the file
    //         for (const auto& ev : addCnt) {
    //             outputFile << ev << std::endl;
    //         }
    //         // Close the file
    //         outputFile.close();
    //         std::cout << "Data saved to file." << std::endl;
    //     } else {
    //         std::cerr << "Failed to open the file." << std::endl;
    //     }
    // }

    
    outputObj["statistics"] = {
            {"simulation_time", durationSimulation.count()},
            {"benchmark", ddsim->getName()},
            {"n_qubits", +ddsim->getNumberOfQubits()},
            {"applied_gates", ddsim->getNumberOfOps()}
    };



    std::vector<dd::fp> time_record;
    time_record = ddsim->getTimeRecord();
    std::ofstream outputFile("../../log/results/time/"+ddsim->getName()+"_DDSIM.txt");
    if (outputFile.is_open()) {
        // Write the vector to the file
        for (const auto& pair : time_record) {
            outputFile << pair << std::endl;
        }
        // Close the file
        outputFile.close();
        std::cout << "Time data saved to file." << std::endl;
    } else {
        std::cerr << "Failed to open the time file." << std::endl;
    }

    std::vector<size_t> node_record;
    node_record = ddsim->getNodeRecord();
    std::ofstream nodeFile("../../log/results/node/"+ddsim->getName()+"_DDSIM"+".txt");
    if (nodeFile.is_open()) {
        // Write the vector to the file
        for (const auto& pair : node_record) {
            nodeFile << pair << std::endl;
        }
        // Close the file
        nodeFile.close();
        std::cout << "Node data saved to file." << std::endl;
    } else {
        std::cerr << "Failed to open the node file." << std::endl;
    }

    std::cout << std::setw(2) << outputObj << std::endl;
}
