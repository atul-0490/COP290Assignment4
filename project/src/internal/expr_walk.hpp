#pragma once

#include <memory>

#include <arrow/api.h>

#include "Expr.hpp"



namespace dfl
{
namespace arx {

std::shared_ptr<arrow::ChunkedArray> eval_on_table(const std::shared_ptr<arrow::Table>& table,const std::shared_ptr<Expr>& expr);

}  // namespace arx
}  // namespace dfl
