#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Expr.hpp"

namespace dfl {

// ---------------------------------------------------------------------------
// Logical plan DAG
// ---------------------------------------------------------------------------
// Each LazyDataFrame operation produces a new LogicalNode whose `children`
// vector points to the upstream node(s) it depends on. `ScanNode` is the
// only leaf. The QueryOptimizer rewrites this DAG before execution.
// ---------------------------------------------------------------------------

struct LogicalNode {
    std::vector<std::shared_ptr<LogicalNode>> children;
    virtual ~LogicalNode() = default;
};

/// Leaf node: reads data from a CSV or Parquet file.
///
/// `projected_columns` (when non-empty) and `row_limit` (when >= 0) are
/// annotations set by the QueryOptimizer (projection / limit pushdown).
/// The executor applies them immediately after the file is read so that
/// Parquet projection and whole-file scans can be trimmed.
struct ScanNode : LogicalNode {
    std::string path;
    bool isParquet = false;
    std::vector<std::string> projected_columns;   // empty = keep all
    int64_t row_limit = -1;                       // -1 = no limit
};

/// Filters rows by a boolean predicate expression.
struct FilterNode : LogicalNode {
    std::shared_ptr<Expr> predicate;
};

/// Projects a set of (possibly computed) columns.
struct SelectNode : LogicalNode {
    std::vector<ExprBuilder> columns;
};

/// Adds or replaces a single column named `name` with the result of `expr`.
struct WithColNode : LogicalNode {
    std::string name;
    ExprBuilder expr;
};

/// Marks the start of a group-by scope. Followed by an AggNode.
struct GroupByNode : LogicalNode {
    std::vector<std::string> keys;
};

/// Computes aggregations, keyed by the nearest upstream GroupByNode (if any).
struct AggNode : LogicalNode {
    std::map<std::string, ExprBuilder> aggMap;
};

/// Joins the main (children[0]) side with `right` on a list of key columns.
struct JoinNode : LogicalNode {
    std::shared_ptr<LogicalNode> right;
    std::vector<std::string> on;
    std::string how; // "inner", "left", "right", "outer"
};

/// Orders rows by one or more columns.
struct SortNode : LogicalNode {
    std::vector<std::string> columns;
    bool ascending = true;
};

/// Takes the first `n` rows.
struct LimitNode : LogicalNode {
    int64_t n = 0;
};

/// Terminal node that writes results to a file.
struct SinkNode : LogicalNode {
    std::string path;
    bool isParquet = false;
};

/// Produce a Graphviz DOT representation of the plan DAG rooted at `root`.
/// The caller can write this string to a .dot file and render it to .png
/// by invoking the `dot` tool (see LazyDataFrame::explain).
std::string renderDotGraph(const std::shared_ptr<LogicalNode>& root);

} // namespace dfl
