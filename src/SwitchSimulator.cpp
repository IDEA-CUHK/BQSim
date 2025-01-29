#include "SwitchSimulator.hpp"
#include "dd/Export.hpp"
#include "dd/Operations.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <queue>
#include <set>
#include <stdexcept>

using CN = dd::ComplexNumbers;

template<class Config>
void SwitchSimulator<Config>::simulate() {
    bool hasNonmeasurementNonUnitary = false;
    bool hasMeasurements             = false;
    bool measurementsLast            = true;
    SwitchSimulator<Config>::dd->n_thread_exp  = SwitchSimulator<Config>::n_thread_exp;


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

template<class Config>
void SwitchSimulator<Config>::nextPath(std::string& s) {
    std::string::reverse_iterator       iter = s.rbegin();
    const std::string::reverse_iterator end  = s.rend();

    int carry = 1;

    while ((carry != 0) && iter != end) {
        const int value = (*iter - '0') + carry;
        carry           = (value / 2);
        *iter           = static_cast<char>('0' + (value % 2));
        ++iter;
    }
    if (carry != 0) {
        s.insert(0, "1");
    }
}

template<class Config>
void SwitchSimulator<Config>::singleShot(const bool ignoreNonUnitaries) {
    singleShots++;
    const auto nQubits = qc->getNqubits();
    SwitchSimulator<Config>::rootEdge = SwitchSimulator<Config>::dd->makeZeroState(static_cast<dd::Qubit>(nQubits));
    SwitchSimulator<Config>::dd->incRef(SwitchSimulator<Config>::rootEdge);
    const size_t nDim    = std::pow(2, nQubits);

    std::size_t                 opNum = 0;
    std::map<std::size_t, bool> classicValues;

    int n_thread = (int)std::pow(2, n_thread_exp);
    std::vector<dd::fp*> z_real_vec(n_thread);
    std::vector<dd::fp*> z_imag_vec(n_thread);
    SwitchSimulator<Config>::EMA_v = nQubits;
    total_macs = 0;
    if (enable_switch == true) {
    if (fuse == 0) {
        std::size_t total_comp = 0;
        // disabled gate merging optimization
        for (auto& op: *qc) {
            if (op->isNonUnitaryOperation()) {
                if (ignoreNonUnitaries) {
                    continue;
                }
                if (auto* nonUnitaryOp = dynamic_cast<qc::NonUnitaryOperation*>(op.get())) {
                    if (op->getType() == qc::Measure) {
                        continue;
                    } else if (nonUnitaryOp->getType() == qc::Reset) {
                        continue;
                    } else if (op->getType() == qc::Barrier) {
                        continue;
                    } else {
                        throw std::runtime_error("Unsupported non-unitary functionality.");
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                if (SwitchSimulator<Config>::switched == false)
                    SwitchSimulator<Config>::dd->garbageCollect();
                else SwitchSimulator<Config>::dd->garbageCollect();
            } else {
                if (op->isClassicControlledOperation()) {
                    if (auto* classicallyControlledOp = dynamic_cast<qc::ClassicControlledOperation*>(op.get())) {
                        throw std::runtime_error("Unsupported classical control functionality.");
                    } else {
                        throw std::runtime_error("Dynamic cast to ClassicControlledOperation failed.");
                    }
                }
                if (opNum % 100 == 0)
                    std::cout << "[Instruction Count]  " << opNum <<std::endl;
                auto start_time1 = std::chrono::system_clock::now();
                // simulation begins
                auto ddOp = dd::getDD(op.get(), SwitchSimulator<Config>::dd);
                if (SwitchSimulator<Config>::switched == false) {
                    auto tmp  = SwitchSimulator<Config>::dd->multiply(ddOp, SwitchSimulator<Config>::rootEdge);
                    SwitchSimulator<Config>::dd->incRef(tmp);
                    SwitchSimulator<Config>::dd->decRef(SwitchSimulator<Config>::rootEdge);
                    SwitchSimulator<Config>::rootEdge = tmp;
                }
                else {
                    SwitchSimulator<Config>::dd->DDArrMultiplyIP(ddOp, 
                        SwitchSimulator<Config>::state_real[opNum%2], SwitchSimulator<Config>::state_imag[opNum%2],
                        SwitchSimulator<Config>::state_real[(opNum+1)%2], SwitchSimulator<Config>::state_imag[(opNum+1)%2],
                        nDim);
                    std::unordered_map<decltype(ddOp.p), std::size_t> mac_map;  
                    total_comp += SwitchSimulator<Config>::dd->DMAVMACStatIP(ddOp, mac_map, nDim, n_thread_exp);
                    mac_map.clear();
                    std::memset(SwitchSimulator<Config>::state_real[opNum%2], 0, sizeof(dd::fp) * nDim);
                    std::memset(SwitchSimulator<Config>::state_imag[opNum%2], 0, sizeof(dd::fp) * nDim);
                    // SwitchSimulator<Config>::dd->makeZeroVector(SwitchSimulator<Config>::stateVec[opNum%2], nDim);
                }
                SwitchSimulator<Config>::dd->garbageCollect();
                // simulation finishes
                auto end_time1 = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds1 = end_time1-start_time1;
                double new_time = elapsed_seconds1.count();
                if (SwitchSimulator<Config>::switched == false)
                    SwitchSimulator<Config>::timeRecord1.push_back(new_time);
                else
                    SwitchSimulator<Config>::timeRecord2.push_back(new_time);
                double new_ddsize = SwitchSimulator<Config>::dd->size(SwitchSimulator<Config>::rootEdge);
                double new_v = SwitchSimulator<Config>::EMA_v * SwitchSimulator<Config>::beta + (1-SwitchSimulator<Config>::beta) * new_ddsize;
                if (SwitchSimulator<Config>::switched == false && SwitchSimulator<Config>::EMA_v > 0 && SwitchSimulator<Config>::EMA_v * SwitchSimulator<Config>::threshold < new_ddsize) // 0.0065364 * nDim, 0.008
                {
                    std::cout<<"Switching from DDSIM to FLATDD!!"<<std::endl;
                    std::cout << "Switching at instr.  " << opNum <<std::endl;
                    auto start_times = std::chrono::system_clock::now();
                    SwitchSimulator<Config>::switched = true;
                    if (SwitchSimulator<Config>::ddsim_convert) getVectorFromDD((opNum+1)%2); else getVectorFromDDSwitch1((opNum+1)%2);
                    // SwitchSimulator<Config>::stateVec[(opNum+1)%2] = 
                    // getVectorFromDDSwitch1((opNum+1)%2);
                    // SwitchSimulator<Config>::stateVec[opNum%2] = SwitchSimulator<Config>::dd->makeZeroVector(nDim);
                    std::memset(SwitchSimulator<Config>::state_real[opNum%2], 0, sizeof(dd::fp) * nDim);
                    std::memset(SwitchSimulator<Config>::state_imag[opNum%2], 0, sizeof(dd::fp) * nDim);
                    auto end_times = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_secondss = end_times-start_times;
                    SwitchSimulator<Config>::switchTime = elapsed_secondss.count();
                    std::cout << "DD->Array conversion time: " << SwitchSimulator<Config>::switchTime << std::endl;
                }
                SwitchSimulator<Config>::EMA_v = new_v;
                // std::cout << "-----------------------------------------------"<<std::endl;
                opNum++;
            }
        }
        std::cout << "Cost: " << total_comp << std::endl;
        SwitchSimulator<Config>::stateVecIdx = opNum%2;
    }
    else {
        for (int i = 0; i < n_thread; i++)
        {
            z_real_vec[i] = (dd::fp*)malloc(sizeof(dd::fp) * nDim);
            z_imag_vec[i] = (dd::fp*)malloc(sizeof(dd::fp) * nDim);
        }

        
        // enabling gate merging optimization
        auto merged_gate = SwitchSimulator<Config>::dd->makeIdent(qc->getNqubits());
        std::vector<decltype(merged_gate)> dd_schedule;
        std::vector<bool> in_or_out;
        int merge_num = 0;
        for (auto& op: *qc) {
            if (op->isNonUnitaryOperation()) {
                if (ignoreNonUnitaries) {
                    continue;
                }
                if (auto* nonUnitaryOp = dynamic_cast<qc::NonUnitaryOperation*>(op.get())) {
                    if (op->getType() == qc::Measure) {
                        continue;
                    } else if (nonUnitaryOp->getType() == qc::Reset) {
                        continue;
                    } else if (op->getType() == qc::Barrier) {
                        continue;
                    } else {
                        throw std::runtime_error("Unsupported non-unitary functionality.");
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                SwitchSimulator<Config>::dd->garbageCollect();
            } else {
                if (op->isClassicControlledOperation()) {
                    if (auto* classicallyControlledOp = dynamic_cast<qc::ClassicControlledOperation*>(op.get())) {
                        throw std::runtime_error("Unsupported classical control functionality.");
                    } else {
                        throw std::runtime_error("Dynamic cast to ClassicControlledOperation failed.");
                    }
                }
                merge_num++;
                // simulation begins
                auto ddOp = dd::getDD(op.get(), SwitchSimulator<Config>::dd);
                merged_gate = SwitchSimulator<Config>::dd->multiply(ddOp, merged_gate);
                int node_count = SwitchSimulator<Config>::dd->size(merged_gate);
                if ( (SwitchSimulator<Config>::switched == false && merge_num > 5)
                    || opNum == qc->ops.size()-1 || opNum < qc->ops.size()-1 && qc->ops[opNum+1]->isNonUnitaryOperation() ) { 
                    auto start_time1 = std::chrono::system_clock::now();
                    if (SwitchSimulator<Config>::switched == false) {
                        auto tmp  = SwitchSimulator<Config>::dd->multiply(merged_gate, SwitchSimulator<Config>::rootEdge);
                        SwitchSimulator<Config>::dd->incRef(tmp);
                        SwitchSimulator<Config>::dd->decRef(SwitchSimulator<Config>::rootEdge);
                        SwitchSimulator<Config>::rootEdge = tmp;
                    }
                    merge_num = 0;
                    auto end_time1 = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_seconds1 = end_time1-start_time1;
                    double new_time = elapsed_seconds1.count();
                    SwitchSimulator<Config>::timeRecord1.push_back(new_time);
                    SwitchSimulator<Config>::dd->garbageCollect();
                    // simulation finishes
                    double new_ddsize = SwitchSimulator<Config>::dd->size(SwitchSimulator<Config>::rootEdge);
                    double new_v = SwitchSimulator<Config>::EMA_v * SwitchSimulator<Config>::beta + (1-SwitchSimulator<Config>::beta) * new_ddsize;

                    if (SwitchSimulator<Config>::switched == false && SwitchSimulator<Config>::EMA_v > 0 && SwitchSimulator<Config>::EMA_v * SwitchSimulator<Config>::threshold < new_ddsize) 
                    {
                        std::cout<<"Switching from DDSIM to FLATDD!!"<<std::endl;
                        std::cout << "Switching at instr. " << opNum <<std::endl;
                        auto start_times = std::chrono::system_clock::now();
                        SwitchSimulator<Config>::switched = true;
                        if (SwitchSimulator<Config>::ddsim_convert) getVectorFromDD(0); else getVectorFromDDSwitch1(0);
                        // SwitchSimulator<Config>::stateVec[opNum%2] = SwitchSimulator<Config>::dd->makeZeroVector(nDim);
                        std::memset(SwitchSimulator<Config>::state_real[1], 0, sizeof(dd::fp) * nDim);
                        std::memset(SwitchSimulator<Config>::state_imag[1], 0, sizeof(dd::fp) * nDim);
                        auto end_times = std::chrono::system_clock::now();
                        std::chrono::duration<double> elapsed_secondss = end_times-start_times;
                        SwitchSimulator<Config>::switchTime = elapsed_secondss.count();
                        std::cout << "DD->Array conversion time: " << SwitchSimulator<Config>::switchTime << std::endl;

                        // DP gate merging scheduling
                        ////
                        auto start = std::chrono::system_clock::now();
                        auto cur_dd = SwitchSimulator<Config>::dd->makeIdent(qc->getNqubits());
                        if (fuse == 1) {
                            std::cout << "Using greedy merge... " << std::endl;
                            std::unordered_map<decltype(cur_dd.p), std::size_t> mac_map; 
                            size_t dp_cost = 0;
                            size_t dp_cost1 = 0;
                            size_t dp_cost2 = 0;
                            bool dp_cache = 0;
                            // size_t dp_wm = 0;
                            // loose ends here
                            auto merge_cur = cur_dd;
                            decltype(merge_cur) merge_nxt;
                            std::size_t total_comp = 0;
                            std::size_t saved_comp = 0;
                            // int weight_comp = 1;
                            for (int nxt_op = opNum+1; nxt_op < qc->ops.size() && !(qc->ops[nxt_op]->isNonUnitaryOperation()); nxt_op++)
                            {
                                auto nxt_dd = dd::getDD(qc->ops[nxt_op].get(), SwitchSimulator<Config>::dd);
                                size_t mac_op = 0;
                                size_t nxt_cost1 = SwitchSimulator<Config>::dd->DMAVMACStatIP(nxt_dd, mac_map, nDim, n_thread_exp);
                                size_t nxt_cost2 = SwitchSimulator<Config>::dd->DMAVMACStatOP1(nxt_dd, mac_map, nDim, n_thread_exp, mac_op);
                                bool nxt_cache = 0;
                                bool merge_cache = 0;
                                size_t nxt_cost = 0;
                                size_t merge_cost = 0;
                                if (nxt_cost1 < nxt_cost2) {
                                    nxt_cache = 0;
                                    nxt_cost = nxt_cost1;
                                } else {
                                    nxt_cache = 1;
                                    nxt_cost = nxt_cost2;
                                }
                                merge_nxt = SwitchSimulator<Config>::dd->multiply(nxt_dd, merge_cur);
                                mac_op = 0;
                                size_t merge_cost1 = SwitchSimulator<Config>::dd->DMAVMACStatIP(merge_nxt, mac_map, nDim, n_thread_exp);
                                size_t merge_cost2 = SwitchSimulator<Config>::dd->DMAVMACStatOP1(merge_nxt, mac_map, nDim, n_thread_exp, mac_op);
                                if (merge_cost1 < merge_cost2) {
                                    merge_cache = 0;
                                    merge_cost = merge_cost1;
                                } else {
                                    merge_cache = 1;
                                    merge_cost = merge_cost2;
                                }
                                if (dp_cost + nxt_cost < merge_cost || nxt_op == qc->ops.size() - 1 || qc->ops[nxt_op+1]->isNonUnitaryOperation())
                                {
                                    // checkout
                                    dd_schedule.push_back(merge_cur);
                                    total_comp += ( dp_cost1);
                                    in_or_out.push_back(dp_cache);
                                    saved_comp += (dp_cost1 - dp_cost);
                                    dp_cost1 = nxt_cost1;
                                    dp_cost2 = nxt_cost2;
                                    dp_cost = nxt_cost;
                                    dp_cache = nxt_cache;

                                    merge_cur = nxt_dd;
                                    mac_map.clear();
                                }
                                else {
                                    merge_cur = merge_nxt;
                                    // dp_mac = merge_mac;
                                    dp_cost1 = merge_cost1;
                                    dp_cost2 = merge_cost2;
                                    dp_cost = merge_cost;
                                    dp_cache = merge_cache;
                                }
                            }
                            dd_schedule.push_back(merge_cur);
                            in_or_out.push_back(0);
                            total_comp += ( dp_cost1);
                            std::cout << "Cost: " << total_comp-saved_comp << std::endl;
                            std::cout << "Saved cost %: " << 100 *(double) saved_comp / (double) total_comp << "%"<< std::endl;
                        }
                        else if (fuse == 2) { // merge dd ops
                            std::cout << "Using op-count-based merge from DATE '19... " << std::endl;
                            size_t merged_num = 0;
                            // loose ends here
                            std::size_t total_comp = 0;
                            auto merge_cur = cur_dd;
                            decltype(merge_cur) merge_nxt;
                            std::unordered_map<decltype(cur_dd.p), std::size_t> mac_map;
                            for (int nxt_op = opNum+1; nxt_op < qc->ops.size() && !(qc->ops[nxt_op]->isNonUnitaryOperation()); nxt_op++)
                            {
                                auto nxt_dd = dd::getDD(qc->ops[nxt_op].get(), SwitchSimulator<Config>::dd);
                                merge_nxt = SwitchSimulator<Config>::dd->multiply(nxt_dd, merge_cur);
                                merged_num++;
                                // int node_count = SwitchSimulator<Config>::dd->size(merge_nxt);
                                if (merged_num > 5)
                                {
                                    // checkout
                                    dd_schedule.push_back(merge_cur);
                                    total_comp += SwitchSimulator<Config>::dd->DMAVMACStatIP(merge_cur, mac_map, nDim, n_thread_exp);
                                    mac_map.clear();
                                    in_or_out.push_back(0);
                                    merge_cur = nxt_dd;
                                    merged_num = 0;
                                }
                                else {
                                    merge_cur = merge_nxt;
                                }
                            }
                            dd_schedule.push_back(merge_cur);
                            in_or_out.push_back(0);
                            std::cout << "Cost: " << total_comp << std::endl;
                        }
                        
                        auto end = std::chrono::system_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                        std::cout << "Gate merging time: "<<  (double)elapsed.count() / 1000 << '\n';
                        break;   
                    }
                    SwitchSimulator<Config>::EMA_v = new_v;
                    merged_gate = SwitchSimulator<Config>::dd->makeIdent(qc->getNqubits());
                }
                opNum++;
            }
        }
        std::cout << "Merged Gate Number: "<< dd_schedule.size() << "\n";
        auto start_time2 = std::chrono::system_clock::now();
        int sid = 0;
        for (sid = 0; sid < dd_schedule.size(); sid++) 
        {
            auto start_time_merge = std::chrono::system_clock::now();
            if (in_or_out[sid] == 1 && enable_cache == 1) {
                SwitchSimulator<Config>::dd->DDArrMultiplyOP(dd_schedule[sid], 
                    SwitchSimulator<Config>::state_real[sid%2], SwitchSimulator<Config>::state_imag[sid%2],
                    SwitchSimulator<Config>::state_real[(sid+1)%2], SwitchSimulator<Config>::state_imag[(sid+1)%2],
                    nDim, z_real_vec, z_imag_vec);
            }
            else 
                SwitchSimulator<Config>::dd->DDArrMultiplyIP(dd_schedule[sid], 
                    SwitchSimulator<Config>::state_real[sid%2], SwitchSimulator<Config>::state_imag[sid%2],
                    SwitchSimulator<Config>::state_real[(sid+1)%2], SwitchSimulator<Config>::state_imag[(sid+1)%2],
                    nDim);
            std::memset(SwitchSimulator<Config>::state_real[sid%2], 0, sizeof(dd::fp) * nDim);
            std::memset(SwitchSimulator<Config>::state_imag[sid%2], 0, sizeof(dd::fp) * nDim);
            auto end_time_merge = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_secs_merge = end_time_merge-start_time_merge;
            SwitchSimulator<Config>::timeRecord2.push_back(elapsed_secs_merge.count());
        }
        auto end_time2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds2 = end_time2-start_time2;
        std::cout<< "runtime after conversion: " << elapsed_seconds2.count() <<std::endl;

        SwitchSimulator<Config>::stateVecIdx = sid%2;
    }
    }
    else {
    for (int i = 0; i < n_thread; i++)
    {
        z_real_vec[i] = (dd::fp*)malloc(sizeof(dd::fp) * nDim);
        z_imag_vec[i] = (dd::fp*)malloc(sizeof(dd::fp) * nDim);
    }

    std::memset(SwitchSimulator<Config>::state_real[0], 0, sizeof(dd::fp) * nDim);
    std::memset(SwitchSimulator<Config>::state_imag[0], 0, sizeof(dd::fp) * nDim);
    if (custom_inputs == false)
        getVectorFromDD(0);
    else
        getVectorCustomInput(0);
    std::memset(SwitchSimulator<Config>::state_real[1], 0, sizeof(dd::fp) * nDim);
    std::memset(SwitchSimulator<Config>::state_imag[1], 0, sizeof(dd::fp) * nDim);
    if (fuse == 0) {
        for (auto& op: *qc) {
            if (op->isNonUnitaryOperation()) {
                if (ignoreNonUnitaries) {
                    continue;
                }
                if (auto* nonUnitaryOp = dynamic_cast<qc::NonUnitaryOperation*>(op.get())) {
                    if (op->getType() == qc::Measure) {
                        continue;
                    } else if (nonUnitaryOp->getType() == qc::Reset) {
                        continue;
                    } else if (op->getType() == qc::Barrier) {
                        continue;
                    } else {
                        throw std::runtime_error("Unsupported non-unitary functionality.");
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                SwitchSimulator<Config>::dd->garbageCollect();
            } else {
                if (op->isClassicControlledOperation()) {
                    if (auto* classicallyControlledOp = dynamic_cast<qc::ClassicControlledOperation*>(op.get())) {
                        throw std::runtime_error("Unsupported classical control functionality.");
                    } else {
                        throw std::runtime_error("Dynamic cast to ClassicControlledOperation failed.");
                    }
                }
                if (opNum % 100 == 0)
                    std::cout << "[Instruction Count]  " << opNum <<std::endl;
                // simulation begins
                auto ddOp = dd::getDD(op.get(), SwitchSimulator<Config>::dd);
                {
                    SwitchSimulator<Config>::dd->DDArrMultiplyIP(ddOp, 
                        SwitchSimulator<Config>::state_real[opNum%2], SwitchSimulator<Config>::state_imag[opNum%2],
                        SwitchSimulator<Config>::state_real[(opNum+1)%2], SwitchSimulator<Config>::state_imag[(opNum+1)%2],
                        nDim);
                    std::memset(SwitchSimulator<Config>::state_real[(opNum)%2], 0, sizeof(dd::fp) * nDim);
                    std::memset(SwitchSimulator<Config>::state_imag[(opNum)%2], 0, sizeof(dd::fp) * nDim);
                }
                SwitchSimulator<Config>::dd->garbageCollect();
                // simulation finishes
                // std::cout << "-----------------------------------------------"<<std::endl;
                opNum++;
            }
        }
        SwitchSimulator<Config>::stateVecIdx = opNum%2;
    }
    else {
        std::vector<dd::mEdge> dd_schedule;
        std::vector<bool> in_or_out;

        std::size_t total_comp = 0;
        std::size_t saved_comp = 0;
        auto start = std::chrono::system_clock::now();
        auto cur_dd = SwitchSimulator<Config>::dd->makeIdent(qc->getNqubits());
        if (fuse == 1) {
            std::cout << "Using greedy merge... " << std::endl;
            std::unordered_map<decltype(cur_dd.p), std::size_t> mac_map; 
            size_t dp_cost = 0;
            size_t dp_cost1 = 0;
            size_t dp_cost2 = 0;
            bool dp_cache = 0;
            // size_t dp_wm = 0;
            // loose ends here
            auto merge_cur = cur_dd;
            decltype(merge_cur) merge_nxt;
            std::size_t total_comp = 0;
            std::size_t saved_comp = 0;
            size_t dp_mac = 0;
            // int weight_comp = 1;
            for (int nxt_op = opNum; nxt_op < qc->ops.size() && !(qc->ops[nxt_op]->isNonUnitaryOperation()); nxt_op++)
            {
                auto nxt_dd = dd::getDD(qc->ops[nxt_op].get(), SwitchSimulator<Config>::dd);
                size_t nxt_mac = 0;
                size_t tmp_mac_op = 0;
                size_t nxt_cost1 = SwitchSimulator<Config>::dd->DMAVMACStatIP(nxt_dd, mac_map, nDim, n_thread_exp);
                size_t nxt_cost2 = SwitchSimulator<Config>::dd->DMAVMACStatOP1(nxt_dd, mac_map, nDim, n_thread_exp, tmp_mac_op);
                bool nxt_cache = 0;
                bool merge_cache = 0;
                size_t nxt_cost = 0;
                size_t merge_cost = 0;
                if (nxt_cost1 < nxt_cost2) {
                    nxt_cache = 0;
                    nxt_cost = nxt_cost1;
                    nxt_mac = nxt_cost1 * std::pow(2, n_thread_exp);
                } else {
                    nxt_cache = 1;
                    nxt_cost = nxt_cost2;
                    nxt_mac = tmp_mac_op;
                }
                merge_nxt = SwitchSimulator<Config>::dd->multiply(nxt_dd, merge_cur);
                tmp_mac_op = 0;
                size_t merge_cost1 = SwitchSimulator<Config>::dd->DMAVMACStatIP(merge_nxt, mac_map, nDim, n_thread_exp);
                size_t merge_cost2 = SwitchSimulator<Config>::dd->DMAVMACStatOP1(merge_nxt, mac_map, nDim, n_thread_exp, tmp_mac_op);
                size_t merge_mac = 0;
                if (merge_cost1 < merge_cost2) {
                    merge_cache = 0;
                    merge_cost = merge_cost1;
                    merge_mac = merge_cost1 * std::pow(2, n_thread_exp);
                } else {
                    merge_cache = 1;
                    merge_cost = merge_cost2;
                    merge_mac = tmp_mac_op;
                }
                if (dp_cost + nxt_cost < merge_cost || nxt_op == qc->ops.size() - 1 || qc->ops[nxt_op+1]->isNonUnitaryOperation())
                {
                    // checkout
                    dd_schedule.push_back(merge_cur);
                    total_comp += ( dp_cost1);
                    total_macs += dp_mac;
                    in_or_out.push_back(dp_cache);
                    saved_comp += (dp_cost1 - dp_cost);
                    dp_cost1 = nxt_cost1;
                    dp_cost2 = nxt_cost2;
                    dp_mac = nxt_mac;
                    dp_cost = nxt_cost;
                    dp_cache = nxt_cache;

                    merge_cur = nxt_dd;
                    mac_map.clear();
                }
                else {
                    merge_cur = merge_nxt;
                    dp_mac = merge_mac;
                    dp_cost1 = merge_cost1;
                    dp_cost2 = merge_cost2;
                    dp_cost = merge_cost;
                    dp_cache = merge_cache;
                }
            }
            dd_schedule.push_back(merge_cur);
            in_or_out.push_back(0);
            total_comp += ( dp_cost1);
            std::cout << "Cost: " << total_comp-saved_comp << std::endl;
            std::cout << "Saved cost %: " << 100 *(double) saved_comp / (double) total_comp << "%"<< std::endl;
        }
        else if (fuse == 2) { // merge dd ops
            std::cout << "Using op-count-based merge from DATE '19... " << std::endl;
            size_t merged_num = 0;
            // loose ends here
            auto merge_cur = cur_dd;
            decltype(merge_cur) merge_nxt;
            for (int nxt_op = opNum; nxt_op < qc->ops.size() && !(qc->ops[nxt_op]->isNonUnitaryOperation()); nxt_op++)
            {
                auto nxt_dd = dd::getDD(qc->ops[nxt_op].get(), SwitchSimulator<Config>::dd);
                merge_nxt = SwitchSimulator<Config>::dd->multiply(nxt_dd, merge_cur);
                merged_num++;
                // int node_count = SwitchSimulator<Config>::dd->size(merge_nxt);
                if (merged_num > 5)
                {
                    // checkout
                    dd_schedule.push_back(merge_cur);
                    in_or_out.push_back(0);
                    merge_cur = nxt_dd;
                    merged_num = 0;
                }
                else {
                    merge_cur = merge_nxt;
                }
            }
            dd_schedule.push_back(merge_cur);
            in_or_out.push_back(0);
        }

        auto end = std::chrono::system_clock::now();
        fused_gate_num = dd_schedule.size();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "Gate merging time: "<<  (double)elapsed.count() / 1000 << '\n';
        // std::cout << "Merged Gate Number: "<< dd_schedule.size() << "\n";
        int sid = 0;
        for (sid = 0; sid < dd_schedule.size(); sid++) 
        {
            if (in_or_out[sid] == 1 && enable_cache == 1) {
                SwitchSimulator<Config>::dd->DDArrMultiplyOP(dd_schedule[sid], 
                    SwitchSimulator<Config>::state_real[sid%2], SwitchSimulator<Config>::state_imag[sid%2],
                    SwitchSimulator<Config>::state_real[(sid+1)%2], SwitchSimulator<Config>::state_imag[(sid+1)%2],
                    nDim, z_real_vec, z_imag_vec);
            }
            else 
                SwitchSimulator<Config>::dd->DDArrMultiplyIP(dd_schedule[sid], 
                    SwitchSimulator<Config>::state_real[sid%2], SwitchSimulator<Config>::state_imag[sid%2],
                    SwitchSimulator<Config>::state_real[(sid+1)%2], SwitchSimulator<Config>::state_imag[(sid+1)%2],
                    nDim);
            std::memset(SwitchSimulator<Config>::state_real[sid%2], 0, sizeof(dd::fp) * nDim);
            std::memset(SwitchSimulator<Config>::state_imag[sid%2], 0, sizeof(dd::fp) * nDim);
        }

        SwitchSimulator<Config>::stateVecIdx = sid%2;
    }
    for (int i = 0; i < n_thread; i++)
    {
        free(z_real_vec[i]);
        free(z_imag_vec[i]);
    }
    }

    /////////////////////////////////////////////////////
}

template class SwitchSimulator<dd::DDPackageConfig>;
