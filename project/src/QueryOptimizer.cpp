#include "QueryOptimizer.hpp"

namespace dfl {

std::shared_ptr<LogicalNode> QueryOptimizer::optimize(
    const std::shared_ptr<LogicalNode>& plan) const {
    return plan;
}

std::shared_ptr<LogicalNode> QueryOptimizer::predicatePushdown(
    const std::shared_ptr<LogicalNode>& node) const {
    return node;
}

std::shared_ptr<LogicalNode> QueryOptimizer::projectionPushdown(
    const std::shared_ptr<LogicalNode>& node) const {
    return node;
}

std::shared_ptr<LogicalNode> QueryOptimizer::constantFolding(
    const std::shared_ptr<LogicalNode>& node) const {
    return node;
}

std::shared_ptr<LogicalNode> QueryOptimizer::expressionSimplification(
    const std::shared_ptr<LogicalNode>& node) const {
    return node;
}

std::shared_ptr<LogicalNode> QueryOptimizer::limitPushdown(
    const std::shared_ptr<LogicalNode>& node) const {
    return node;
}

} // namespace dfl
