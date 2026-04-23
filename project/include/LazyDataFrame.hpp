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

/// @brief Deferred-execution DataFrame backed by a LogicalNode DAG.
///
/// Each operation returns a new `LazyDataFrame` whose plan extends the
/// parent's plan. No actual data is processed until `collect()` (or a
/// sink / explain) is called; at that point the `QueryOptimizer`
/// transforms the plan and a physical executor materialises the result
/// into an `EagerDataFrame`.
///
/// @code
///   auto result = scan_csv("sales.csv")
///                    .filter(col("region") == lit("US"))
///                    .group_by({"store"})
///                    .aggregate({{"total", col("price").sum()}})
///                    .collect();
/// @endcode
class LazyDataFrame {
public:
    /// @brief Construct an empty (no-plan) LazyDataFrame.
    LazyDataFrame();

    /// @brief Wrap an existing logical plan root.
    /// @param plan Non-null root of a logical plan DAG.
    explicit LazyDataFrame(std::shared_ptr<LogicalNode> plan);

    // Defaulted rule-of-5 members — the only state is a shared_ptr so
    // moves and copies are both cheap.
    LazyDataFrame(const LazyDataFrame&)                = default;
    LazyDataFrame(LazyDataFrame&&) noexcept            = default;
    LazyDataFrame& operator=(const LazyDataFrame&)     = default;
    LazyDataFrame& operator=(LazyDataFrame&&) noexcept = default;
    ~LazyDataFrame()                                    = default;

    // --- Selection / projection ---

    /// @brief Project columns by name. @copydetails EagerDataFrame::select
    /// @return A new LazyDataFrame with a `SelectNode` appended.
    LazyDataFrame select(const std::vector<std::string>& columns) const;

    /// @brief Project a set of computed columns given as expressions.
    /// @param exprs Expressions to evaluate on the upstream frame.
    /// @return A new LazyDataFrame with a `SelectNode` appended.
    LazyDataFrame select(const std::vector<ExprBuilder>& exprs) const;

    /// @brief Convenience overload — `lf.select({"a","b"})`.
    ///
    /// Disambiguates braced-init between the string-vector and
    /// ExprBuilder-vector overloads (same trick as `EagerDataFrame`).
    LazyDataFrame select(std::initializer_list<const char*> columns) const;

    // --- Row operations ---

    /// @brief Filter rows using a boolean predicate (deferred).
    /// @param predicate Boolean ExprBuilder.
    /// @return A new LazyDataFrame with a `FilterNode` appended.
    LazyDataFrame filter(const ExprBuilder& predicate) const;

    /// @brief Add / overwrite a column (deferred).
    /// @param name  Output column name.
    /// @param expr  Expression whose result becomes the column.
    /// @return A new LazyDataFrame with a `WithColNode` appended.
    LazyDataFrame with_column(const std::string& name,
                              const ExprBuilder& expr) const;

    // --- Grouping / aggregation ---

    /// @brief Declare a group-by (deferred).
    /// @param keys Ordered group-key column names.
    /// @return A new LazyDataFrame with a `GroupByNode` appended.
    LazyDataFrame group_by(const std::vector<std::string>& keys) const;

    /// @brief Declare aggregations to run on the current (possibly
    ///        grouped) frame.
    /// @param aggMap Map of output column name → aggregation expression.
    /// @return A new LazyDataFrame with an `AggNode` appended.
    LazyDataFrame aggregate(const std::map<std::string, ExprBuilder>& aggMap) const;

    // --- Joins / ordering / limits ---

    /// @brief Equi-join with `other` (deferred).
    /// @param other Right-hand lazy frame.
    /// @param on    Join-key column names.
    /// @param how   Join flavour: `"inner" | "left" | "right" | "outer"`.
    /// @return A new LazyDataFrame with a `JoinNode` appended.
    /// @throws std::invalid_argument on an unknown `how`.
    LazyDataFrame join(const LazyDataFrame& other,
                       const std::vector<std::string>& on,
                       const std::string& how = "inner") const;

    /// @brief Sort rows (deferred).
    /// @param columns   Sort keys in priority order.
    /// @param ascending `true` for ascending, `false` for descending.
    /// @return A new LazyDataFrame with a `SortNode` appended.
    LazyDataFrame sort(const std::vector<std::string>& columns,
                       bool ascending = true) const;

    /// @brief Keep only the first `n` rows (deferred).
    /// @param n Number of rows to keep (negatives treated as 0).
    /// @return A new LazyDataFrame with a `LimitNode` appended.
    LazyDataFrame head(int64_t n) const;

    // --- Sinks ---

    /// @brief Execute the plan and write the result to `path` as CSV.
    /// @param path Destination filesystem path.
    /// @throws std::runtime_error on any I/O error.
    void sink_csv(const std::string& path) const;

    /// @brief Execute the plan and write the result to `path` as Parquet.
    /// @param path Destination filesystem path.
    /// @throws std::runtime_error on any I/O error.
    void sink_parquet(const std::string& path) const;

    // --- Terminal operations ---

    /// @brief Optimise the plan (via `QueryOptimizer`) and execute it.
    /// @return A materialised `EagerDataFrame`.
    /// @throws std::runtime_error on plan-level errors (missing columns,
    ///         type mismatches, invalid join modes, …).
    EagerDataFrame collect() const;

    /// @brief Execute the plan WITHOUT running the `QueryOptimizer`.
    ///
    /// Primarily used by benchmarks that want to quantify optimizer
    /// speedups. The result is equivalent (up to row ordering for
    /// operations that make no stability guarantees) to `collect()`.
    ///
    /// @return A materialised `EagerDataFrame`.
    EagerDataFrame collect_raw() const;

    /// @brief Render the optimised plan DAG to `pngPath` via Graphviz.
    ///
    /// Non-fatal if `dot` is unavailable — in that case the intermediate
    /// `.dot` source is preserved and its path is printed to stderr so
    /// the user can render it by hand.
    ///
    /// @param pngPath Destination `.png` path.
    void explain(const std::string& pngPath) const;

    /// @brief Read-only access to the current (un-optimised) plan root.
    /// @return The plan root pointer (may be null for default ctor).
    std::shared_ptr<LogicalNode> plan() const;

private:
    std::shared_ptr<LogicalNode> plan_;
};

} // namespace dfl
