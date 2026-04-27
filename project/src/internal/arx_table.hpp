#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "Expr.hpp"



namespace dfl
{
namespace arx {

inline std::string synthesize_column_label(const std::shared_ptr<Expr>& e, size_t index) {
    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e)) return a->alias;
    if (auto c = std::dynamic_pointer_cast<ColExpr>(e)) return c->name;
    return "expr_" + std::to_string(index);
}

inline std::shared_ptr<arrow::Table> bundle_columns(
    const std::vector<std::string>& names,
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& cols) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        fields.push_back(arrow::field(names[i], cols[i]->type()));
    }
    return arrow::Table::Make(arrow::schema(fields), cols);
}

}  // namespace arx
}  // namespace dfl
