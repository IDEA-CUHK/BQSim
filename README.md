# BQSim: GPU-accelerated Batch Quantum Circuit Simulation using Decision Diagram

This Artifact describes the steps to compile and run `BQSim` on various quantum circuits and provides a performance comparison of `BQSim` against `cuQuantum`, `Qiskit Aer`, and `FlatDD`, reproducing the overall comparison results from Section 4.2. 

The artifact includes the source code for `BQSim`, `cuQuantum`, `Qiskit Aer`, and `FlatDD`, along with 16 quantum circuits and their corresponding inputs with varying numbers of qubits for evaluation. Running the artifact requires a CUDA-enabled GPU with at least 48 GB of memory, 20 GB of system RAM, and 2 GB of free disk space.

## Artifact check-list (meta-information)

* Compilation: GCC 12.3.0, NVCC 12.6, CMake 3.22.1. We provide an automated compilation script.
* Datasets: 16 quantum circuits in `circuits/`, and the corresponding random circuit inputs.
* Hardware: CUDA-enabled GPU with 48 GB of memory and 20 GB of system RAM.
* Metrics: Simulation runtime.
* Output: Metric data in the console log. 
* Experiments: We provide automated scripts for running the experiments.
* Disk space: 2 GB.
* How much time is needed to prepare workflow (approximately)?: 10 minutes.
* How much time is needed to complete experiments (approximately)?: All the experiments take approximately four days to complete, as demonstrated in Section 4.2. To reduce runtime, we can evaluate each simulator separately. The fastest simulator, `BQSim`, takes less than 20 minutes, whereas the slowest simulator, `FlatDD`, takes more than two days.

## Description

1. Hardware dependencies: An x86 host with a CUDA-enabled GPU (at least 48 GB of memory), 20 GB of system RAM, and 2 GB of free disk space.

2. Software dependencies: Our experiments are conducted on a Ubuntu 22.04.3 LTS machine with the following software dependencies:

* CUDA 12.6 with cuQuantum SDK.
* GCC 12.3.0, NVCC 12.6, CMake 3.22.1.
* libeigen3-dev.
* OpenMP.
* Python 3.10.12 with NumPy 1.26.4, Qiskit Aer 0.15.0, and Qiskit 1.2.0.

3. Datasets: The artifact includes 16 `MQT-Bench` quantum circuits along with their corresponding randomly generated inputs.

## Compilation

Run the compilation script compile.sh, which will automatically generate the executables for `BQSim` and the baseline simulators (`cuQuantum` and `FlatDD`, `Qiskit Aer` is provided as a Python file).

`~/BQSim$ ./compile.sh`

## Experiment workflow

After compilation, we execute the automated scripts to run `BQSim`, `cuQuantum`, `Qiskit Aer`, and `FlatDD` on 16 quantum circuits. The scripts report the runtime of each simulator for each circuit.

## Evaluation and expected results

To execute simulators `BQSim`, `cuQuantum`, `Qiskit Aer`, and `FlatDD` on 16 quantum circuits, we run the `overall.sh` script. Since the output may contain many lines, it is redirected to a log file, `overall.txt`, located in `log/outputs/`.

`~/BQSim$ ./compile.sh > log/outputs/overall.txt`

However, this process takes approximately four days. To reduce the runtime, we can evaluate each simulator separately by running individual scripts:

* `BQSim`: `bqsim.sh`
* `cuQuantum`: `cuquantum.sh`
* `Qiskit Aer`: `qiskit-aer.sh`
* `FlatDD`: `flatdd.sh`

The fastest simulator, BQSim, takes less than 20 minutes, whereas the slowest simulator, FlatDD, takes more than two days. Each simulator can be run independently with the following commands, and the simulation runtime will be recorded in respective log files:

`~/BQSim$ ./bqsim.sh > log/outputs/bqsim.txt`

`~/BQSim$ ./cuquantum.sh > log/outputs/cuquantum.txt`

`~/BQSim$ ./qiskit-aer.sh > log/outputs/qiskit-aer.txt`

`~/BQSim$ ./flatdd.sh > log/outputs/flatdd.txt`

### How to read the log files:

* `BQSim`: Look for the field `"simulation_time":`.
* `cuQuantum`: Look for the field `cuQuantum runtime:`.
* `Qiskit Aer` and `FlatDD`: Look for the field `============Execution time in ms:`.

# Reference
+ [Advanced simulation of quantum computations](https://ieeexplore.ieee.org/abstract/document/8355954)
+ [Quantum++: A modern C++ quantum computing library](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0208073)