#include "TypeUtils.hpp"

namespace dfl {

ColType promoteTypes(ColType a, ColType b) {
    // Stub: in step 1 we just echo back the first operand.
    (void)b;
    return a;
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

ColType arrowTypeToColType(const std::shared_ptr<arrow::DataType>& /*t*/) {
    return ColType::INT64;
}

} // namespace dfl
