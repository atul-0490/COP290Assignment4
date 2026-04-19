#include "EagerDataFrame.hpp"

#include <iostream>

namespace dfl {

EagerDataFrame::EagerDataFrame() : table_(nullptr) {}

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> table)
    : table_(std::move(table)) {}

// --- DataFrame interface ---

std::vector<std::string> EagerDataFrame::columnNames() const {
    std::vector<std::string> names;
    if (!table_) return names;
    for (const auto& field : table_->schema()->fields()) {
        names.push_back(field->name());
    }
    return names;
}

std::vector<ColType> EagerDataFrame::columnTypes() const {
    std::vector<ColType> types;
    if (!table_) return types;
    for (const auto& field : table_->schema()->fields()) {
        types.push_back(arrowTypeToColType(field->type()));
    }
    return types;
}

int64_t EagerDataFrame::numRows() const {
    return table_ ? table_->num_rows() : 0;
}

void EagerDataFrame::print(int64_t maxRows) const {
    if (!table_) {
        std::cout << "(empty EagerDataFrame)\n";
        return;
    }
    const auto& schema = table_->schema();
    for (int i = 0; i < schema->num_fields(); ++i) {
        std::cout << schema->field(i)->name();
        if (i + 1 < schema->num_fields()) std::cout << " | ";
    }
    std::cout << "\n";

    const int64_t rows = std::min<int64_t>(maxRows, table_->num_rows());
    for (int64_t r = 0; r < rows; ++r) {
        for (int c = 0; c < schema->num_fields(); ++c) {
            auto col = table_->column(c);
            auto scalar_res = col->GetScalar(r);
            if (scalar_res.ok()) {
                std::cout << (*scalar_res)->ToString();
            } else {
                std::cout << "?";
            }
            if (c + 1 < schema->num_fields()) std::cout << " | ";
        }
        std::cout << "\n";
    }
}

// --- Operations (stubs) ---

EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& /*columns*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::select(const std::vector<ExprBuilder>& /*exprs*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::filter(const ExprBuilder& /*predicate*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::with_column(const std::string& /*name*/,
                                           const ExprBuilder& /*expr*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::group_by(const std::vector<std::string>& /*keys*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::aggregate(
    const std::map<std::string, ExprBuilder>& /*aggMap*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& /*other*/,
                                    const std::vector<std::string>& /*on*/,
                                    const std::string& /*how*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& /*columns*/,
                                    bool /*ascending*/) const {
    return EagerDataFrame(table_);
}

EagerDataFrame EagerDataFrame::head(int64_t /*n*/) const {
    return EagerDataFrame(table_);
}

// --- I/O (stubs) ---

void EagerDataFrame::write_csv(const std::string& /*path*/) const {}
void EagerDataFrame::write_parquet(const std::string& /*path*/) const {}

std::shared_ptr<arrow::Table> EagerDataFrame::table() const { return table_; }

std::shared_ptr<arrow::ChunkedArray> EagerDataFrame::evalExpr(
    const std::shared_ptr<Expr>& /*expr*/) const {
    return nullptr;
}

} // namespace dfl
