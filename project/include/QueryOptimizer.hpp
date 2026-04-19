#pragma once

#include <memory>

#include "LogicalPlan.hpp"

namespace dfl {

/// Rule-based query optimiser for LazyDataFrame plans.
///
/// `optimize()` runs all rewrite passes in a fixed, correctness-preserving
/// order and returns the rewritten plan root. Each private member is a
/// single pass that walks the DAG bottom-up and returns a new node (the
/// original plan is never mutated in place).
class QueryOptimizer {
public:
    std::shared_ptr<LogicalNode> optimize(
        const std::shared_ptr<LogicalNode>& plan) const;

private:
    std::shared_ptr<LogicalNode> predicatePushdown(
        const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> projectionPushdown(
        const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> constantFolding(
        const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> expressionSimplification(
        const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> limitPushdown(
        const std::shared_ptr<LogicalNode>& node) const;
};

} // namespace dfl
