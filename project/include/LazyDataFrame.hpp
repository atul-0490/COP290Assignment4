#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "EagerDataFrame.hpp"
#include "Expr.hpp"
#include "LogicalPlan.hpp"

namespace dfl {

/// LazyDataFrame — deferred execution backed by a LogicalNode DAG.
///
/// Each operation returns a new LazyDataFrame whose plan extends the parent's
/// plan. No actual data is processed until `collect()` (or a sink/explain) is
/// called; at that point the QueryOptimizer transforms the plan and a
/// physical executor materialises the result into an `EagerDataFrame`.
class LazyDataFrame {
public:
    LazyDataFrame();
    explicit LazyDataFrame(std::shared_ptr<LogicalNode> plan);

    // --- Selection / projection ---
    LazyDataFrame select(const std::vector<std::string>& columns) const;
    LazyDataFrame select(const std::vector<ExprBuilder>& exprs) const;

    /// Disambiguates `lf.select({"a","b"})` the same way EagerDataFrame does
    /// (see EagerDataFrame::select(std::initializer_list<const char*>)).
    LazyDataFrame select(std::initializer_list<const char*> columns) const;

    // --- Row operations ---
    LazyDataFrame filter(const ExprBuilder& predicate) const;
    LazyDataFrame with_column(const std::string& name,
                              const ExprBuilder& expr) const;

    // --- Grouping / aggregation ---
    LazyDataFrame group_by(const std::vector<std::string>& keys) const;
    LazyDataFrame aggregate(const std::map<std::string, ExprBuilder>& aggMap) const;

    // --- Joins / ordering / limits ---
    LazyDataFrame join(const LazyDataFrame& other,
                       const std::vector<std::string>& on,
                       const std::string& how = "inner") const;
    LazyDataFrame sort(const std::vector<std::string>& columns,
                       bool ascending = true) const;
    LazyDataFrame head(int64_t n) const;

    // --- Sinks ---
    void sink_csv(const std::string& path) const;
    void sink_parquet(const std::string& path) const;

    // --- Terminal operations ---
    /// Optimise the plan and execute it, returning a materialised frame.
    EagerDataFrame collect() const;

    /// Dump the (optimised) plan DAG to the given .png path using Graphviz.
    void explain(const std::string& pngPath) const;

    /// Access the current (un-optimised) plan root.
    std::shared_ptr<LogicalNode> plan() const;

private:
    std::shared_ptr<LogicalNode> plan_;
};

} // namespace dfl
