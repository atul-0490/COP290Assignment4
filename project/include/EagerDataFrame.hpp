#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>

#include "DataFrame.hpp"
#include "Expr.hpp"



namespace dfl
{

class EagerDataFrame : public DataFrame {
public:
    EagerDataFrame();

    explicit EagerDataFrame(std::shared_ptr<arrow::Table> table);

    EagerDataFrame(const EagerDataFrame&) = default;
    EagerDataFrame(EagerDataFrame&&) noexcept = default;
    EagerDataFrame& operator=(const EagerDataFrame&) = default;
    EagerDataFrame& operator=(EagerDataFrame&&) noexcept = default;
    ~EagerDataFrame() override = default;


    std::vector<std::string> columnNames() const override;
    std::vector<ColType> columnTypes() const override;
    int64_t numRows() const override;
    int64_t num_rows() const;
    int64_t num_columns() const;
    void print(int64_t maxRows = 20) const override;

    EagerDataFrame select(const std::vector<std::string>& columns) const;

    EagerDataFrame select(const std::vector<ExprBuilder>& exprs) const;

    EagerDataFrame select(std::initializer_list<const char*> columns) const;
    EagerDataFrame filter(const ExprBuilder& predicate) const;

    EagerDataFrame with_column(const std::string& name, const ExprBuilder& expr) const;
    EagerDataFrame group_by(const std::vector<std::string>& keys) const;
    EagerDataFrame aggregate(const std::map<std::string, ExprBuilder>& aggMap) const;

    EagerDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const;

    EagerDataFrame join(const EagerDataFrame& other, const std::vector<std::string>& on, const std::string& how = "inner") const;

    EagerDataFrame sort(const std::vector<std::string>& columns, bool ascending = true) const;
    EagerDataFrame head(int64_t n) const;

    void write_csv(const std::string& path) const;

    void write_parquet(const std::string& path) const;

    std::shared_ptr<arrow::Table> table() const;

    std::shared_ptr<arrow::ChunkedArray> evalExpr( const std::shared_ptr<Expr>& expr) const;

private:
    std::shared_ptr<arrow::Table> table_;
    std::vector<std::string> group_keys_;
};
} 
