#include "TypeUtils.hpp"

#include <stdexcept>
#include <string>

#include <arrow/api.h>
#include <arrow/type.h>

namespace dfl {

namespace {

// Order numeric types from narrowest to widest so we can compute the wider
// of two promotable types with a simple max().
int numericRank(ColType t) {
    switch (t) {
        case ColType::INT32:   return 1;
        case ColType::INT64:   return 2;
        case ColType::FLOAT32: return 3;
        case ColType::FLOAT64: return 4;
        default:               return 0;
    }
}

[[noreturn]] void throwIncompatible(ColType a, ColType b) {
    throw std::invalid_argument(
        "Incompatible types: " + colTypeToString(a) + " and " + colTypeToString(b));
}

} // namespace

ColType promoteTypes(ColType a, ColType b) {
    if (a == b) return a;

    const bool aNum = isNumeric(a);
    const bool bNum = isNumeric(b);

    // int + float → float; otherwise widest numeric wins.
    if (aNum && bNum) {
        const bool aFloat = (a == ColType::FLOAT32 || a == ColType::FLOAT64);
        const bool bFloat = (b == ColType::FLOAT32 || b == ColType::FLOAT64);
        if (aFloat || bFloat) {
            // Pick the widest float seen; if only one side is float, we still
            // widen to at least float64 to keep precision predictable.
            if (a == ColType::FLOAT64 || b == ColType::FLOAT64) return ColType::FLOAT64;
            return ColType::FLOAT32;
        }
        // Both are int: pick the wider integer.
        return numericRank(a) >= numericRank(b) ? a : b;
    }

    // Any mix of numeric / string / boolean is an error.
    throwIncompatible(a, b);
}

bool isNumeric(ColType t) {
    switch (t) {
        case ColType::INT32:
        case ColType::INT64:
        case ColType::FLOAT32:
        case ColType::FLOAT64:
            return true;
        default:
            return false;
    }
}

std::string colTypeToString(ColType t) {
    switch (t) {
        case ColType::INT32:   return "int32";
        case ColType::INT64:   return "int64";
        case ColType::FLOAT32: return "float32";
        case ColType::FLOAT64: return "float64";
        case ColType::STRING:  return "string";
        case ColType::BOOLEAN: return "boolean";
    }
    return "unknown";
}

ColType arrowTypeToColType(const std::shared_ptr<arrow::DataType>& t) {
    if (!t) throw std::runtime_error("arrowTypeToColType: null DataType");

    switch (t->id()) {
        case arrow::Type::INT32:   return ColType::INT32;
        case arrow::Type::INT64:   return ColType::INT64;
        case arrow::Type::FLOAT:   return ColType::FLOAT32;
        case arrow::Type::DOUBLE:  return ColType::FLOAT64;
        case arrow::Type::STRING:  return ColType::STRING;
        case arrow::Type::LARGE_STRING: return ColType::STRING;
        case arrow::Type::BOOL:    return ColType::BOOLEAN;
        default:
            throw std::runtime_error(
                "Unsupported Arrow type: " + t->ToString());
    }
}

} // namespace dfl
