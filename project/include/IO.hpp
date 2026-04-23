#pragma once

#include <map>
#include <memory>
#include <string>

#include <arrow/api.h>

#include "EagerDataFrame.hpp"
#include "LazyDataFrame.hpp"

namespace dfl {

// ---------------------------------------------------------------------------
// Eager I/O — immediately materialises data into an EagerDataFrame.
// ---------------------------------------------------------------------------

/// @brief Load a CSV file into a materialised EagerDataFrame.
///
/// Uses Arrow's CSV reader with default `ReadOptions`, `ParseOptions`,
/// and `ConvertOptions` (comma separator, header on first line, types
/// auto-inferred).
///
/// @param path Filesystem path to the CSV file.
/// @return     A fresh EagerDataFrame wrapping the parsed table.
/// @throws     std::runtime_error if the file cannot be opened or parsed.
///
/// Example:
/// @code
///   auto df = dfl::read_csv("/tmp/people.csv");
///   df.print();
/// @endcode
EagerDataFrame read_csv(const std::string& path);

/// @brief Load a Parquet file into a materialised EagerDataFrame.
///
/// @param path Filesystem path to the `.parquet` file.
/// @return     A fresh EagerDataFrame wrapping the decoded table.
/// @throws     std::runtime_error on any Arrow/Parquet error, or if the
///             library was built without Parquet support.
EagerDataFrame read_parquet(const std::string& path);

// ---------------------------------------------------------------------------
// Lazy I/O — returns a LazyDataFrame whose plan contains only a ScanNode.
// ---------------------------------------------------------------------------

/// @brief Create a LazyDataFrame whose plan reads `path` as CSV.
///
/// No I/O actually happens here — the read is deferred until
/// `LazyDataFrame::collect()` is invoked. The optimizer may rewrite the
/// scan (projection pushdown, limit pushdown) before execution.
///
/// @param path Filesystem path to the CSV file.
/// @return     A LazyDataFrame wrapping a single ScanNode.
LazyDataFrame scan_csv(const std::string& path);

/// @brief Create a LazyDataFrame whose plan reads `path` as Parquet.
///
/// @param path Filesystem path to the `.parquet` file.
/// @return     A LazyDataFrame wrapping a single ScanNode.
LazyDataFrame scan_parquet(const std::string& path);

// ---------------------------------------------------------------------------
// Build an EagerDataFrame directly from a map of name → Arrow Array.
// All arrays must have the same length.
// ---------------------------------------------------------------------------

/// @brief Build an EagerDataFrame from an in-memory map of Arrow arrays.
///
/// Useful when test fixtures or call sites need to feed synthetic data to
/// the library without round-tripping through a CSV/Parquet file. All
/// arrays must have identical length — the function checks this and
/// throws if any column's length differs.
///
/// @param columns Map of column name → non-null `arrow::Array` pointer.
/// @return        A fresh EagerDataFrame containing those columns.
/// @throws        std::invalid_argument if any two arrays have different
///                lengths, or if any array pointer is null.
EagerDataFrame from_columns(
    const std::map<std::string, std::shared_ptr<arrow::Array>>& columns);

} // namespace dfl
