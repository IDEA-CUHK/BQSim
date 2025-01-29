#ifndef DDFLAT_DDSIMULATOR_H
#define DDFLAT_DDSIMULATOR_H

#include "QuantumComputation.hpp"
#include "Definitions.hpp"
#include "dd/Package.hpp"
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

template<class Config = dd::DDPackageConfig>
class DDSimulator {
public:
    explicit DDSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_) : qc(std::move(qc_)) {
        DDSimulator<Config>::dd->resize(qc->getNqubits());
    };

    virtual ~DDSimulator() = default;

    void simulate(bool collapsAllGates);

    template<class ReturnType = dd::ComplexValue>
    [[nodiscard]] std::vector<std::pair<dd::fp, dd::fp>> getVector() const {
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
            results[i]                = {cv.r, cv.i};
            nextPath(path);
            sum += cv.r * cv.r + cv.i * cv.i;
        }
        // printf("sum = %lf\n", sum);
        return results;
    }

    [[nodiscard]] std::vector<dd::fp> getTimeRecord() const {
        return timeRecord;
    }

    [[nodiscard]] std::vector<size_t> getMul() const {
        return mulCnt;
    }

    [[nodiscard]] std::vector<size_t> getAdd() const {
        return addCnt;
    }
    [[nodiscard]] std::vector<size_t> getNodeRecord() const {
        return nodeCnt;
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


    std::unique_ptr<dd::Package<Config>> dd = std::make_unique<dd::Package<Config>>();
    dd::vEdge                            rootEdge{};
    bool                                    collapsAllGates;
    std::vector<dd::fp>                     timeRecord;
    std::vector<size_t>                     mulCnt;
    std::vector<size_t>                     addCnt;
    std::vector<size_t>                     nodeCnt;
    unsigned int                            n_thread_exp;

protected:
    dd::fp        epsilon = 0.001;
    std::unique_ptr<qc::QuantumComputation> qc;
    std::size_t                             singleShots{0};

    void singleShot(bool ignoreNonUnitaries);
    static void nextPath(std::string& s);
};


#endif //DDFLAT_DDSIMULATOR_H
