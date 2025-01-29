#ifndef SWITCH_SIMULATOR_H
#define SWITCH_SIMULATOR_H

#include "QuantumComputation.hpp"
#include "Definitions.hpp"
#include "dd/SwitchPackage.hpp"
#include "operations/OpType.hpp"

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

template<class Config = dd::DDPackageConfig>
class SwitchSimulator {
public:
    explicit SwitchSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_) : qc(std::move(qc_)) {
        SwitchSimulator<Config>::dd->resize(qc->getNqubits());
        const auto nQubits = qc->getNqubits();
        const size_t nDim    = std::pow(2, nQubits);
        state_real0 = (dd::fp *)malloc(sizeof(dd::fp) * nDim);
        state_real1 = (dd::fp *)malloc(sizeof(dd::fp) * nDim);
        state_imag0 = (dd::fp *)malloc(sizeof(dd::fp) * nDim);
        state_imag1 = (dd::fp *)malloc(sizeof(dd::fp) * nDim);
        memset(state_real0, 0, sizeof(dd::fp) * nDim);
        memset(state_real1, 0, sizeof(dd::fp) * nDim);
        memset(state_imag0, 0, sizeof(dd::fp) * nDim);
        memset(state_imag1, 0, sizeof(dd::fp) * nDim);
        state_real.push_back(state_real0);
        state_real.push_back(state_real1);
        state_imag.push_back(state_imag0);
        state_imag.push_back(state_imag1);
    };

    ~SwitchSimulator() {
        free(state_real[0]);
        free(state_real[1]);
        free(state_imag[0]);
        free(state_imag[1]);
    }

    void simulate();


    [[nodiscard]]
    void getVector(dd::fp* &real_vec, dd::fp* &imag_vec) const {
        dd::fp sum = 0.0;
        if (getNumberOfQubits() >= 60) {
            // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
            throw std::range_error("getVector only supports less than 60 qubits.");
        }
        real_vec = state_real[stateVecIdx];
        imag_vec = state_imag[stateVecIdx];
    }

   template<class ReturnType = dd::ComplexValue>
    [[nodiscard]] void getVectorFromDD(int idx) const {
        dd::fp sum = 0.0;
        if (getNumberOfQubits() >= 60) {
            // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
            throw std::range_error("getVector only supports less than 60 qubits.");
        }
        std::string             path(getNumberOfQubits(), '0');
        std::vector<ReturnType> results;
        results.resize(1ULL << getNumberOfQubits());
        for (std::size_t i = 0; i < 1ULL << getNumberOfQubits(); ++i) {
            const std::string      correctedPath{path.rbegin(), path.rend()};
            const dd::ComplexValue cv = dd->getValueByPath(rootEdge, correctedPath);
            state_real[idx][i]               = cv.r;
            state_imag[idx][i]               = cv.i;
            nextPath(path);
        }
    }

    [[nodiscard]] void getVectorCustomInput(int idx) const {
        std::string filename = "../../input_batch/n"+std::to_string(qc->getNqubits())+".txt";
        std::ifstream file;
        file.open((filename).c_str());

        if (!file.is_open()) {
            std::cerr << "Failed to open file." << std::endl;
            exit(-1);
        }
        std::string line;
        while (getline(file, line)) {
            std::istringstream iss(line);
            dd::fp real, imag;
            int amp_id = 0;
            while (iss >> real >> imag) {
                state_real[idx][amp_id] = real;
                state_imag[idx][amp_id] = imag;
                amp_id++;
            }
        }
        file.close();
    }

    // template<class ReturnType = dd::ComplexValue>
    // [[nodiscard]] void getVectorFromDDSwitch(int idx) const {
    //     dd::fp sum = 0.0;
    //     if (getNumberOfQubits() >= 60) {
    //         // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
    //         throw std::range_error("getVector only supports less than 60 qubits.");
    //     }
    //     std::string             path(getNumberOfQubits()-n_thread_exp, '0');
    //     int n_thread = std::pow(2, n_thread_exp);
    //     std::vector<std::thread> threads;
    //     std::vector<std::vector<int>> left_or_right(n_thread, std::vector<int>(n_thread_exp));
    //     for (size_t i = 0; i < n_thread; i++)
    //     {
    //         int idx = i;
    //         for (size_t j = 0; j < n_thread_exp; j++) {
    //             left_or_right[i][j] = idx / (int)std::pow(2, n_thread_exp-1-j);
    //             idx = idx % (int)std::pow(2, n_thread_exp-1-j);
    //         }
    //     }


    //     for (size_t i = 0; i < n_thread; i++) {
    //         bool go_on = true;
    //         decltype(rootEdge) this_edge = rootEdge;
    //         for (size_t j = 0; j < n_thread_exp; j++) {

    //             int dir = left_or_right[i][j];
    //             this_edge = this_edge.p->e[dir];
    //             if (this_edge.isZeroTerminal()) {
    //                 go_on = false;
    //                 break;
    //             }
    //         }
    //         if (go_on) {
    //             auto size_i = SwitchSimulator<Config>::dd->size(this_edge);
    //             std::cout << "thread " << i << ": "<<size_i <<" "<< this_edge.p<< std::endl;
    //         }
    //     }

    //     auto size_i = SwitchSimulator<Config>::dd->size(rootEdge);
    //     std::cout << "DD size : "<<size_i << std::endl;
    //     std::cout << "2^n/t: " << std::pow(2, getNumberOfQubits())/n_thread << std::endl;

    //     for (size_t i = 0; i < n_thread; i++) {
    //         threads.emplace_back([this, i, left_or_right, path, idx](){
    //             size_t segment = (size_t)std::pow(2, getNumberOfQubits()-n_thread_exp);
    //             size_t offset = i* segment;
    //             // dd::fp w_real = 1;
    //             // dd::fp w_imag = 0;

    //             std::string outer_path = "";
    //             bool go_on = true;
    //             decltype(rootEdge) this_edge = rootEdge;
    //             for (size_t j = 0; j < n_thread_exp; j++) {
    //                 int dir = left_or_right[i][j];
    //                 this_edge = this_edge.p->e[dir];
    //                 if (this_edge.isZeroTerminal()) {
    //                     go_on = false;
    //                     break;
    //                 }
    //                 outer_path += std::to_string(dir);

    //             }

    //             if (go_on) {
    //                 std::reverse(outer_path.begin(), outer_path.end());
    //                 std::string thread_path = path + outer_path;
    //                 std::reverse(thread_path.begin(), thread_path.end());
    //                 for (std::size_t j = offset; j < offset+segment; ++j) {
    //                     const std::string      correctedPath{thread_path.rbegin(), thread_path.rend()};
    //                     const dd::ComplexValue cv = dd->getValueByPathPar(rootEdge, correctedPath);
    //                     state_real[idx][j]               = cv.r;
    //                     state_imag[idx][j]               = cv.i;
    //                     nextPath(thread_path);
    //                 }
    //             }
    //         });
    //     }
    //     for (auto &thread : threads)
    //     {
    //         thread.join();
    //     }
    // }

    template<class ReturnType = dd::ComplexValue>
    [[nodiscard]] void getVectorFromDDSwitch1(int idx) const {
        // auto size_i = SwitchSimulator<Config>::dd->size(rootEdge);
        // std::cout << "DD size : "<<size_i << std::endl;


        dd::fp sum = 0.0;
        if (getNumberOfQubits() >= 60) {
            // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
            throw std::range_error("getVector only supports less than 60 qubits.");
        }
        
        int n_thread = std::pow(2, n_thread_exp);
        std::vector<std::thread> threads;

        std::map<std::string, decltype(rootEdge)> last_paths; 
        std::map<std::string, decltype(rootEdge)> this_paths;
        std::map<std::string, size_t> last_offsets; 
        std::map<std::string, size_t> this_offsets;

        last_paths[""] = rootEdge;  
        last_offsets[""] = 0;  
        int seg_exp = getNumberOfQubits();

        struct SegStruct
        {
            decltype(rootEdge) edge;
            dd::fp real;
            dd::fp imag;
            size_t beg;
            size_t end;
        };
        // all the segments to transfer later
        std::map<std::string, SegStruct> last_segs;
        std::map<std::string, SegStruct> this_segs;
        last_segs[""] = SegStruct{rootEdge, dd::RealNumber::val(rootEdge.w.r), dd::RealNumber::val(rootEdge.w.i), 0, std::pow(2, seg_exp)};
        auto thisNode = rootEdge.p;
        bool ir_flag = false;

        // dealing with initial regularities
        while (thisNode->e[0].p == thisNode->e[1].p)
        {
            ir_flag = true;
            for (auto elem : last_segs)
            {
                auto this_edge = elem.second.edge;
                if (!this_edge.p->e[0].w.exactlyZero())
                {
                    SegStruct seg_inst;

                    seg_inst.edge = this_edge.p->e[0];
                    seg_inst.real = elem.second.real * dd::RealNumber::val(this_edge.p->e[0].w.r) - elem.second.imag * dd::RealNumber::val(this_edge.p->e[0].w.i);
                    seg_inst.imag = elem.second.real * dd::RealNumber::val(this_edge.p->e[0].w.i) + elem.second.imag * dd::RealNumber::val(this_edge.p->e[0].w.r);
                    seg_inst.beg = elem.second.beg;
                    seg_inst.end = elem.second.beg + std::pow(2, seg_exp-1);

                    this_segs[elem.first+"0"] = seg_inst;
                    thisNode = this_edge.p->e[0].p;
                }
                if (!this_edge.p->e[1].w.exactlyZero())
                {

                    SegStruct seg_inst;

                    seg_inst.edge = this_edge.p->e[1];
                    seg_inst.real = elem.second.real * dd::RealNumber::val(this_edge.p->e[1].w.r) - elem.second.imag * dd::RealNumber::val(this_edge.p->e[1].w.i);
                    seg_inst.imag = elem.second.real * dd::RealNumber::val(this_edge.p->e[1].w.i) + elem.second.imag * dd::RealNumber::val(this_edge.p->e[1].w.r);
                    seg_inst.beg = elem.second.beg + std::pow(2, seg_exp-1);
                    seg_inst.end = elem.second.end;

                    this_segs[elem.first+"1"] = seg_inst;
                    thisNode = this_edge.p->e[1].p;
                }
            }
            if (this_segs.size() > n_thread) {
                break;
            }
            else {
                seg_exp--;
                last_segs = this_segs;
                this_segs.clear();
            }
        }
        
        SegStruct first_seg;
        if (ir_flag)
        {
            auto first_elem = last_segs.begin();
            last_paths.clear();
            last_offsets.clear();
            last_paths[first_elem->first] = first_elem->second.edge;
            last_offsets[first_elem->first] = first_elem->second.beg;
            first_seg = first_elem->second;
            last_segs.erase(first_elem->first); 
        }
        
        while (seg_exp >= 1)
        {
            for (auto elem : last_paths)
            {
                auto this_edge = elem.second;
                if (!this_edge.p->e[0].w.exactlyZero())
                {
                    this_paths[elem.first+"0"] = this_edge.p->e[0];
                    this_offsets[elem.first+"0"] = last_offsets[elem.first];
                }
                if (!this_edge.p->e[1].w.exactlyZero())
                {
                    this_paths[elem.first+"1"] = this_edge.p->e[1];
                    this_offsets[elem.first+"1"] = last_offsets[elem.first] + std::pow(2, seg_exp-1);
                }
            }
            if (this_paths.size() > n_thread) {
                break;
            }
            else {
                seg_exp--;
                last_paths = this_paths;
                last_offsets = this_offsets;
                this_paths.clear();
                this_offsets.clear();
            }
        }
        std::string             path(seg_exp, '0');

        for (auto elem : last_paths) {
            threads.emplace_back([this, elem, last_paths, last_offsets, path, idx, seg_exp](){
                size_t segment = (size_t)std::pow(2, seg_exp);
                size_t offset = last_offsets.at(elem.first);

                std::string outer_path = elem.first;
                decltype(rootEdge) this_edge = elem.second;


                std::reverse(outer_path.begin(), outer_path.end());
                std::string thread_path = path + outer_path;
                std::reverse(thread_path.begin(), thread_path.end());
                for (std::size_t j = offset; j < offset+segment; ++j) {
                    const std::string      correctedPath{thread_path.rbegin(), thread_path.rend()};
                    const dd::ComplexValue cv = dd->getValueByPathPar(rootEdge, correctedPath);
                    state_real[idx][j]               = cv.r;
                    state_imag[idx][j]               = cv.i;
                    nextPath(thread_path);
                }
            });
        }
        for (auto &thread : threads)
        {
            thread.join();
        }

        // dealing with the left-over initial regularities
        threads.clear();
        if (ir_flag) {
            for (auto elem : last_segs) {
                threads.emplace_back([this, elem, idx, first_seg](){
                    dd::fp common_r = (elem.second.real * first_seg.real + elem.second.imag * first_seg.imag) / (first_seg.real*first_seg.real+first_seg.imag*first_seg.imag);
                    dd::fp common_i = (elem.second.imag * first_seg.real - elem.second.real * first_seg.imag) / (first_seg.real*first_seg.real+first_seg.imag*first_seg.imag);

                    const __m256d commonr4 = _mm256_set1_pd(common_r);
                    const __m256d commoni4 = _mm256_set1_pd(common_i);

                    for ( int k = 0; k < ( (elem.second.end-elem.second.beg)  & ~0x3); k+= 4 ) {
                        const __m256d zr4   = _mm256_loadu_pd( &(state_real[idx][first_seg.beg+k]) );
                        const __m256d zi4   = _mm256_loadu_pd( &(state_imag[idx][first_seg.beg+k]) );
                        
                        __m256d zr4_m_commonr4 = _mm256_mul_pd(zr4, commonr4);
                        __m256d zi4_m_commoni4 = _mm256_mul_pd(zi4, commoni4);
                        __m256d zr4_m_commoni4 = _mm256_mul_pd(zr4, commoni4);
                        __m256d zi4_m_commonr4 = _mm256_mul_pd(zi4, commonr4);

                        __m256d resr4 = _mm256_add_pd( zr4_m_commonr4, -zi4_m_commoni4 );
                        __m256d resi4 = _mm256_add_pd( zr4_m_commoni4, zi4_m_commonr4 );

                        _mm256_storeu_pd( &(state_real[idx][elem.second.beg+k]), resr4 );
                        _mm256_storeu_pd( &(state_imag[idx][elem.second.beg+k]), resi4 );
                    }
                });
            }
            for (auto &thread : threads)
            {
                thread.join();
            }
        }
    }

    [[nodiscard]] std::vector<dd::fp> getTimeRecord1() const {
        return timeRecord1;
    }
    [[nodiscard]] std::vector<dd::fp> getTimeRecord2() const {
        return timeRecord2;
    }
    [[nodiscard]] size_t getMacs() const {
        return total_macs;
    }
    [[nodiscard]] int getFusedGates() const {
        return fused_gate_num;
    }
    // getSwitchTime
    [[nodiscard]] dd::fp getSwitchTime() const {
        return switchTime;
    }
    [[nodiscard]] std::size_t getNumberOfQubits() const { return qc->getNqubits(); };

    [[nodiscard]] std::size_t getNumberOfOps() const { return qc->getNops(); };

    [[nodiscard]] std::string getName() const { return qc->getName(); };

    [[nodiscard]] static inline std::string toBinaryString(const std::size_t value, const std::size_t numberOfQubits) {
        std::string binary(numberOfQubits, '0');
        for (std::size_t j = 0; j < numberOfQubits; ++j) {
            if ((value & (1U << j)) != 0U) {
                binary[j] = '1';
            }
        }
        return binary;
    }

    void setTolerance(const dd::fp tolerance) {
        dd->cn.setTolerance(tolerance);
    }
    [[nodiscard]] dd::fp getTolerance() const {
        return dd::RealNumber::eps;
    }

    std::unique_ptr<dd::SwitchPackage<Config>>     dd  = std::make_unique<dd::SwitchPackage<Config>>();
    dd::fp *state_real0;
    dd::fp *state_real1;
    dd::fp *state_imag0;
    dd::fp *state_imag1;
    std::vector<dd::fp*> state_real;
    std::vector<dd::fp*> state_imag;
    // std::vector<std::vector<std::pair<dd::fp, dd::fp>>>   stateVec{std::vector<std::pair<dd::fp, dd::fp>>(0), std::vector<std::pair<dd::fp, dd::fp>>(0)};
    int                                     stateVecIdx;
    dd::vEdge                               rootEdge{};
    std::vector<dd::fp>                     timeRecord1;
    std::vector<dd::fp>                     timeRecord2;
    dd::fp                                  switchTime;
    unsigned int                            n_thread_exp;
    bool                                    switched = false;
    bool                                    ddsim_convert = false;
    bool                                    enable_switch = true;
    unsigned int                            fuse = 0;

    bool enable_cache = true;
    double EMA_v = 0;
    double beta = 0.9;
    double threshold = 3.5;
    bool custom_inputs = false;
    

protected:
    dd::fp        epsilon = 0.001;
    std::unique_ptr<qc::QuantumComputation> qc;
    std::size_t                             singleShots{0};

    void singleShot(bool ignoreNonUnitaries);
    static void nextPath(std::string& s);

    std::size_t total_macs;
    int fused_gate_num;
};


#endif //SWITCH_SIMULATOR_H
