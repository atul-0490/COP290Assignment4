#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include <arrow/api.h>
#include <arrow/type.h>



namespace dfl
{

enum class ColType {
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STRING,
    BOOLEAN
};

ColType promoteTypes(ColType a, ColType b);

bool isNumeric(ColType t);

std::string colTypeToString(ColType t);

ColType arrowTypeToColType(const std::shared_ptr<arrow::DataType>& t);

} 
