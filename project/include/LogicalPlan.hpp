#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Expr.hpp"

namespace dfl {


struct LogicalNode {
    std::vector<std::shared_ptr<LogicalNode>> children;
    virtual ~LogicalNode() = default;
};

struct ScanNode : LogicalNode {
    std::string path;                 
    bool isParquet = false;    
    std::vector<std::string> projected_columns;    
    int64_t row_limit = -1;       
};

struct FilterNode : LogicalNode {
    std::shared_ptr<Expr> predicate;
};

struct SelectNode : LogicalNode {
    std::vector<ExprBuilder> columns;
};

struct WithColNode : LogicalNode {
    std::string name;
    ExprBuilder expr;
};

struct GroupByNode : LogicalNode {
    std::vector<std::string> keys;
};

struct AggNode : LogicalNode {
    std::map<std::string, ExprBuilder> aggMap;
};

struct JoinNode : LogicalNode {
    std::shared_ptr<LogicalNode> right;
    std::vector<std::string> on;
    std::string how; 
};

struct SortNode : LogicalNode {
    std::vector<std::string> columns;
    bool ascending = true;
};

struct LimitNode : LogicalNode {
    int64_t n = 0;
};

struct SinkNode : LogicalNode {
    std::string path;
    bool isParquet = false;
};

std::string renderDotGraph(const std::shared_ptr<LogicalNode>& root);

} 
