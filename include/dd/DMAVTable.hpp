#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <malloc.h>

namespace dd {

struct CachedSegment {
  size_t beg_idx;
  size_t end_idx;

  fp real_factor;
  fp imag_factor;
};

template <std::size_t NBUCKET = 512>
class DMAVTable {
public:
  DMAVTable() = default;

  struct Entry {
    mNode* leftOperand;

    CachedSegment result = {0, 0, -1, -1};
  };

  static constexpr std::size_t MASK = NBUCKET - 1;

  static std::size_t hash(const mNode* p) {
    static constexpr std::size_t MASK = NBUCKET - 1;
    std::size_t key = 0;
    for (std::size_t i = 0; i < p->e.size(); ++i) {
      key = combineHash(key, std::hash<Edge<mNode>>{}(p->e[i]));
    }
    key &= MASK;
    return key;
  }

  [[nodiscard]] const auto& getTable() const { return table; }
  void insert(mNode* leftOperand,
              CachedSegment result) {
    const auto key = hash(leftOperand);
    table[key] = Entry{leftOperand, result};
    ++count;
  }

  CachedSegment lookup(const mNode* leftOperand) {
    lookups++;
    const auto key = hash(leftOperand);
    auto& entry = table[key];
    if (entry.leftOperand != leftOperand) {
      return CachedSegment{0, 0, -1, -1};
    }

    hits++;
    return entry.result;
  }

  void clear() {
    if (count > 0) {
      std::fill(table.begin(), table.end(), Entry{});
      count = 0;
    }
  }
  [[nodiscard]] std::size_t getHits() const { return hits; }
  [[nodiscard]] std::size_t getLookups() const { return lookups; }
  [[nodiscard]] fp hitRatio() const {
    if (lookups == 0U) {
      return 1.;
    }
    return static_cast<fp>(getHits()) / static_cast<fp>(getLookups());
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) {
    os << "hits: " << hits << ", looks: " << lookups
       << ", ratio: " << hitRatio() << std::endl;
    return os;
  }

private:
  std::array<Entry, NBUCKET> table{};
  // compute table lookup statistics
  std::size_t hits = 0;
  std::size_t lookups = 0;
  std::size_t count = 0;
  std::size_t collision = 0;
};

}