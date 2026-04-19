#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include <arrow/api.h>
#include <arrow/type.h>

namespace dfl {

/// Strict, immutable column types supported by DataFrameLib.
enum class ColType {
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STRING,
    BOOLEAN
};

/// Compute the promoted result type for a binary operation.
/// Rules:
///   - identical types → same type
///   - int + float    → float (wider of the two)
///   - incompatible   → throws std::invalid_argument
ColType promoteTypes(ColType a, ColType b);

/// True if the type is numeric (int32, int64, float32, float64).
bool isNumeric(ColType t);

/// Human-readable string form of a ColType, useful for error messages.
std::string colTypeToString(ColType t);

/// Map an Arrow DataType to a DataFrameLib ColType. Throws if unsupported.
ColType arrowTypeToColType(const std::shared_ptr<arrow::DataType>& t);

} // namespace dfl
