#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "TypeUtils.hpp"

namespace dfl {

/// Abstract interface common to EagerDataFrame and LazyDataFrame.
///
/// Note: lazy frames advertise their schema without materialising the data,
/// so `numRows()` may return -1 for a LazyDataFrame whose plan has not yet
/// been collected.
class DataFrame {
public:
    virtual ~DataFrame() = default;

    virtual std::vector<std::string> columnNames() const = 0;
    virtual std::vector<ColType>     columnTypes() const = 0;
    virtual int64_t                  numRows()    const = 0;

    /// Pretty-print up to `maxRows` rows of the frame to stdout.
    virtual void print(int64_t maxRows = 20) const = 0;
};

} // namespace dfl
