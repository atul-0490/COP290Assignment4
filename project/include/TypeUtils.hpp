#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include <arrow/api.h>
#include <arrow/type.h>

namespace dfl {

/// @brief Strict, immutable column types supported by DataFrameLib.
///
/// Every column in a DataFrame has exactly one of these types for its
/// entire lifetime. Missing values are represented as nulls inside Arrow
/// arrays — never as sentinel values like NaN.
enum class ColType {
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STRING,
    BOOLEAN
};

/// @brief Compute the result type of a binary arithmetic / comparison
///        operation on operands of types `a` and `b`.
///
/// Promotion rules:
///   - identical types → the same type.
///   - any int + any float → FLOAT64 (widens to double precision).
///   - INT32 + INT64 → INT64.
///   - STRING / BOOLEAN combined with a numeric → throws.
///
/// @param a Left-hand column type.
/// @param b Right-hand column type.
/// @return  The promoted ColType.
/// @throws  std::invalid_argument if `a` and `b` are incompatible.
///
/// Example:
/// @code
///   auto t = dfl::promoteTypes(dfl::ColType::INT32, dfl::ColType::FLOAT64);
///   // t == ColType::FLOAT64
/// @endcode
ColType promoteTypes(ColType a, ColType b);

/// @brief True iff `t` is one of INT32, INT64, FLOAT32, FLOAT64.
/// @param t The column type to classify.
/// @return  `true` for numeric types, `false` for STRING / BOOLEAN.
bool isNumeric(ColType t);

/// @brief Render a ColType as a human-readable string (e.g. `"int32"`,
///        `"float64"`, `"string"`).
///
/// Primarily used when constructing exception messages for type errors.
///
/// @param t The column type.
/// @return  Lowercase string form of the type.
std::string colTypeToString(ColType t);

/// @brief Map an Arrow runtime data type to the matching ColType.
///
/// Supports Int32, Int64, Float, Double, String (Utf8), and Boolean. Any
/// other Arrow type triggers an exception — DataFrameLib deliberately
/// rejects unsupported column types at load time rather than silently
/// coercing them.
///
/// @param t A non-null `arrow::DataType` pointer.
/// @return  The corresponding ColType.
/// @throws  std::runtime_error for unsupported Arrow types.
ColType arrowTypeToColType(const std::shared_ptr<arrow::DataType>& t);

} // namespace dfl
