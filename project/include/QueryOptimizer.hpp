#pragma once

#include <memory>

#include "LogicalPlan.hpp"

namespace dfl {

/// @brief Rule-based query optimiser for LazyDataFrame plans.
///
/// `optimize()` runs all rewrite passes to a fixed point and returns the
/// rewritten plan root. Each private member is a single pass that walks
/// the DAG bottom-up and returns a new node — the original plan is never
/// mutated in place, so callers can reuse and inspect both the
/// unoptimised and optimised DAGs (this is exactly what the benchmark
/// test does).
///
/// The five rules implemented are:
///   1. Constant folding              — e.g. `lit(3)+lit(4) → lit(7)`.
///   2. Expression simplification     — `x*1 → x`, `~~x → x`, etc.
///   3. Predicate pushdown            — push filters below joins and
///                                      group-bys when safe.
///   4. Projection pushdown           — annotate scans with only the
///                                      columns ultimately required.
///   5. Limit pushdown                — absorb `LIMIT` into scans and
///                                      swap it with row-preserving
///                                      nodes.
///
/// @code
///   LazyDataFrame lf = scan_csv("big.csv").filter(col("a") > lit(5));
///   QueryOptimizer opt;
///   auto plan = opt.optimize(lf.plan());
/// @endcode
class QueryOptimizer {
public:
    /// @brief Apply every rewrite rule, iterating until the plan stops
    ///        changing (or a safety cap of 10 iterations is reached).
    ///
    /// @param plan Non-null root of a logical plan.
    /// @return     A new plan that is semantically equivalent to `plan`
    ///             but may execute faster.
    /// @throws     std::runtime_error on malformed plans.
    std::shared_ptr<LogicalNode> optimize(
        const std::shared_ptr<LogicalNode>& plan) const;

private:
    /// @brief Pass that moves `FilterNode`s closer to the source across
    ///        joins and group-bys when the predicate only references
    ///        columns available on that side / in the group keys.
    std::shared_ptr<LogicalNode> predicatePushdown(
        const std::shared_ptr<LogicalNode>& node) const;

    /// @brief Pass that propagates the set of required columns top-down
    ///        and records it in `ScanNode::projected_columns`.
    std::shared_ptr<LogicalNode> projectionPushdown(
        const std::shared_ptr<LogicalNode>& node) const;

    /// @brief Pass that evaluates any fully-constant sub-expression at
    ///        plan-construction time and replaces it with a `LitExpr`.
    std::shared_ptr<LogicalNode> constantFolding(
        const std::shared_ptr<LogicalNode>& node) const;

    /// @brief Pass that rewrites trivially reducible algebraic patterns
    ///        (`x+0 → x`, `x*1 → x`, `~~x → x`, …).
    std::shared_ptr<LogicalNode> expressionSimplification(
        const std::shared_ptr<LogicalNode>& node) const;

    /// @brief Pass that swaps `LimitNode` with row-preserving nodes
    ///        (`SelectNode`, `WithColNode`) and absorbs it into scans.
    std::shared_ptr<LogicalNode> limitPushdown(
        const std::shared_ptr<LogicalNode>& node) const;
};

} // namespace dfl
