#include "DDSimulator.hpp"
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
void DDSimulator<Config>::simulate(bool collapsAllGates) {
    bool hasNonmeasurementNonUnitary = false;
    bool hasMeasurements             = false;
    bool measurementsLast            = true;
    DDSimulator<Config>::collapsAllGates = collapsAllGates;
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
void DDSimulator<Config>::nextPath(std::string& s) {
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
void DDSimulator<Config>::singleShot(const bool ignoreNonUnitaries) {
    singleShots++;
    const auto nQubits = qc->getNqubits();
    DDSimulator<Config>::rootEdge = DDSimulator<Config>::dd->makeZeroState(static_cast<dd::Qubit>(nQubits));
    DDSimulator<Config>::dd->incRef(DDSimulator<Config>::rootEdge);

    std::size_t                 opNum = 0;
    std::map<std::size_t, bool> classicValues;

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
            DDSimulator<Config>::dd->garbageCollect();
        } else {
            if (op->isClassicControlledOperation()) {
                if (auto* classicallyControlledOp = dynamic_cast<qc::ClassicControlledOperation*>(op.get())) {
                    // const auto   startIndex    = static_cast<std::uint16_t>(classicallyControlledOp->getParameter().at(0));
                    // const auto   length        = static_cast<std::uint16_t>(classicallyControlledOp->getParameter().at(1));
                    // const auto   expectedValue = classicallyControlledOp->getExpectedValue();
                    // unsigned int actualValue   = 0;
                    // for (std::size_t i = 0; i < length; i++) {
                    //     actualValue |= (classicValues[startIndex + i] ? 1U : 0U) << i;
                    // }

                    // //std::clog << "expected " << expected_value << " and actual value was " << actual_value << "\n";

                    // if (actualValue != expectedValue) {
                    //     continue;
                    // }
                    throw std::runtime_error("Unsupported classical control functionality.");
                } else {
                    throw std::runtime_error("Dynamic cast to ClassicControlledOperation failed.");
                }
            }
            /*std::clog << "[INFO] op " << op_num << " is " << op->getName() << " on " << +op->getTargets().at(0)
                      << " #controls=" << op->getControls().size()
                      << " statesize=" << dd->size(stateVec) << "\n";//*/
            if (opNum % 100 == 0)
                std::cout << "[Instruction Count]  " << opNum <<std::endl;
            auto start_time1 = std::chrono::system_clock::now();
            auto ddOp = dd::getDD(op.get(), DDSimulator<Config>::dd);
            auto tmp  = DDSimulator<Config>::dd->multiply(ddOp, DDSimulator<Config>::rootEdge);
            // DDSimulator<Config>::mulCnt.push_back(DDSimulator<Config>::dd->mul_cnt);
            // DDSimulator<Config>::addCnt.push_back(DDSimulator<Config>::dd->add_cnt);
            DDSimulator<Config>::nodeCnt.push_back(DDSimulator<Config>::dd->size(tmp));
            if (DDSimulator<Config>::collapsAllGates)
                DDSimulator<Config>::getVector<std::pair<dd::fp, dd::fp>>();
            DDSimulator<Config>::dd->incRef(tmp);
            DDSimulator<Config>::dd->decRef(DDSimulator<Config>::rootEdge);
            DDSimulator<Config>::rootEdge = tmp;
            DDSimulator<Config>::dd->garbageCollect();
            auto end_time1 = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds1 = end_time1-start_time1;
            DDSimulator<Config>::timeRecord.push_back(elapsed_seconds1.count());

            if (elapsed_seconds1.count() > 20) {
                std::cout << "    [Instruction Count]  " << opNum <<  ": " <<elapsed_seconds1.count() << std::endl;
            }

            opNum++;
        }
    }

    return;
}

template class DDSimulator<dd::DDPackageConfig>;
