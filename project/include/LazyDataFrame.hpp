#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "EagerDataFrame.hpp"
#include "Expr.hpp"
#include "LogicalPlan.hpp"

namespace dfl {

class LazyDataFrame {
public:
    LazyDataFrame();

    explicit LazyDataFrame(std::shared_ptr<LogicalNode> plan);

    LazyDataFrame(const LazyDataFrame&) = default;
    LazyDataFrame(LazyDataFrame&&) noexcept = default;
    LazyDataFrame& operator=(const LazyDataFrame&) = default;
    LazyDataFrame& operator=(LazyDataFrame&&) noexcept = default;
    ~LazyDataFrame() = default;


    LazyDataFrame select(const std::vector<std::string>& columns) const;

    LazyDataFrame select(const std::vector<ExprBuilder>& exprs) const;

    LazyDataFrame select(std::initializer_list<const char*> columns) const;

    LazyDataFrame filter(const ExprBuilder& predicate) const;

    LazyDataFrame with_column(const std::string& name, const ExprBuilder& expr) const;

    LazyDataFrame group_by(const std::vector<std::string>& keys) const;

    LazyDataFrame aggregate(const std::map<std::string, ExprBuilder>& aggMap) const;

    LazyDataFrame aggregate( const std::vector<std::pair<std::string, std::string>>& aggs) const;

    LazyDataFrame join(const LazyDataFrame& other, const std::vector<std::string>& on, const std::string& how = "inner") const;

    LazyDataFrame sort(const std::vector<std::string>& columns, bool ascending = true) const;

    LazyDataFrame head(int64_t n) const;

    void sink_csv(const std::string& path) const;

    void sink_parquet(const std::string& path) const;

    EagerDataFrame collect() const;

    EagerDataFrame collect_raw() const;

    void explain(const std::string& pngPath) const;

    std::shared_ptr<LogicalNode> plan() const;

private:
    std::shared_ptr<LogicalNode> plan_;
};
} 
