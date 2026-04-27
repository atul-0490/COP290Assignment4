#pragma once

#include <memory>

#include "LogicalPlan.hpp"



namespace dfl
{

class QueryOptimizer {
public:
    std::shared_ptr<LogicalNode> optimize( const std::shared_ptr<LogicalNode>& plan) const;

private:
    std::shared_ptr<LogicalNode> predicatePushdown( const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> projectionPushdown( const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> constantFolding(const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> expressionSimplification(const std::shared_ptr<LogicalNode>& node) const;

    std::shared_ptr<LogicalNode> limitPushdown(const std::shared_ptr<LogicalNode>& node) const;
};
} 
