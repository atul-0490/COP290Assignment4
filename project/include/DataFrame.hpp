#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "TypeUtils.hpp"

namespace dfl {

/// @brief Abstract interface common to EagerDataFrame and LazyDataFrame.
///
/// `DataFrame` exposes schema introspection (column names / types / row
/// count) and a `print()` helper so code can be written generically
/// against either execution mode. Eager frames always return a concrete
/// `numRows()`; lazy frames may return `-1` when the row count cannot be
/// determined without executing the plan.
class DataFrame {
public:
    virtual ~DataFrame() = default;

    /// @brief Ordered list of column names in this frame.
    /// @return A vector of column names in physical order.
    virtual std::vector<std::string> columnNames() const = 0;

    /// @brief Ordered list of column types, aligned with `columnNames()`.
    /// @return A vector of ColType values, one per column.
    virtual std::vector<ColType> columnTypes() const = 0;

    /// @brief Number of materialised rows in the frame.
    ///
    /// For an EagerDataFrame this is always the concrete row count. For
    /// a LazyDataFrame whose plan has not been collected this may return
    /// `-1` to signal "unknown until executed".
    ///
    /// @return The row count, or `-1` for un-collected lazy frames.
    virtual int64_t numRows() const = 0;

    /// @brief Pretty-print up to `maxRows` rows of the frame to stdout.
    ///
    /// Columns are separated by ` | `; nulls render as the literal text
    /// `"null"`. When the frame has more rows than `maxRows`, an
    /// ellipsis line `... (N more rows)` follows the printed block.
    ///
    /// @param maxRows Maximum number of data rows to print (default 20).
    virtual void print(int64_t maxRows = 20) const = 0;
};

} // namespace dfl
