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
EagerDataFrame read_csv(const std::string& path);
EagerDataFrame read_parquet(const std::string& path);

// ---------------------------------------------------------------------------
// Lazy I/O — returns a LazyDataFrame whose plan contains only a ScanNode.
// ---------------------------------------------------------------------------
LazyDataFrame scan_csv(const std::string& path);
LazyDataFrame scan_parquet(const std::string& path);

// ---------------------------------------------------------------------------
// Build an EagerDataFrame directly from a map of name → Arrow Array.
// All arrays must have the same length.
// ---------------------------------------------------------------------------
EagerDataFrame from_columns(
    const std::map<std::string, std::shared_ptr<arrow::Array>>& columns);

} // namespace dfl
