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

/// @brief Base class for every node in the logical plan DAG.
///
/// Subclasses carry operator-specific fields (predicate, sort keys, etc.).
/// Parents hold their children via `shared_ptr` so the optimizer can
/// share and reuse subtrees across rewrites safely.
struct LogicalNode {
    std::vector<std::shared_ptr<LogicalNode>> children;
    virtual ~LogicalNode() = default;
};

/// @brief Leaf node: reads data from a CSV or Parquet file.
///
/// `projected_columns` (when non-empty) and `row_limit` (when >= 0) are
/// annotations written in by the QueryOptimizer during projection- and
/// limit-pushdown. The executor applies them immediately after the file
/// is read — this is how Parquet column pruning and limited scans are
/// realised.
struct ScanNode : LogicalNode {
    std::string              path;                 ///< File path.
    bool                     isParquet = false;    ///< true → Parquet, false → CSV.
    std::vector<std::string> projected_columns;    ///< Empty = keep all columns.
    int64_t                  row_limit = -1;       ///< -1 = no row limit.
};

/// @brief Filters rows by a boolean predicate expression (`WHERE`).
struct FilterNode : LogicalNode {
    std::shared_ptr<Expr> predicate;
};

/// @brief Projects a set of (possibly computed) columns (`SELECT`).
struct SelectNode : LogicalNode {
    std::vector<ExprBuilder> columns;
};

/// @brief Adds or replaces a single column named `name` with the result
///        of `expr` (Polars-style `with_column`).
struct WithColNode : LogicalNode {
    std::string name;
    ExprBuilder expr;
};

/// @brief Marks the start of a group-by scope. Must be followed by an
///        AggNode to produce output rows.
struct GroupByNode : LogicalNode {
    std::vector<std::string> keys;
};

/// @brief Computes one or more aggregations, keyed by the nearest
///        upstream GroupByNode (if any).
struct AggNode : LogicalNode {
    std::map<std::string, ExprBuilder> aggMap;
};

/// @brief Joins the main input (`children[0]`) with `right` on a list of
///        key columns. `how` is one of `"inner" | "left" | "right" | "outer"`.
struct JoinNode : LogicalNode {
    std::shared_ptr<LogicalNode> right;
    std::vector<std::string>     on;
    std::string                  how; ///< "inner", "left", "right", or "outer".
};

/// @brief Orders rows by one or more columns.
struct SortNode : LogicalNode {
    std::vector<std::string> columns;
    bool                     ascending = true;
};

/// @brief Takes the first `n` rows (`LIMIT`).
struct LimitNode : LogicalNode {
    int64_t n = 0;
};

/// @brief Terminal node that writes results to a file (CSV or Parquet).
struct SinkNode : LogicalNode {
    std::string path;
    bool        isParquet = false;
};

/// @brief Produce a Graphviz DOT representation of the plan DAG rooted
///        at `root`.
///
/// The caller can write this string to a `.dot` file and render it to
/// `.png` by invoking the `dot` tool — exactly what
/// `LazyDataFrame::explain()` does internally.
///
/// @param root A non-null root of a logical plan.
/// @return     DOT source code as a string.
std::string renderDotGraph(const std::shared_ptr<LogicalNode>& root);

} // namespace dfl
