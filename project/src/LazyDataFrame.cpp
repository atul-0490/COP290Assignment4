#include "LazyDataFrame.hpp"

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "IO.hpp"
#include "LogicalPlan.hpp"
#include "QueryOptimizer.hpp"

namespace dfl {


LazyDataFrame::LazyDataFrame() : plan_(nullptr) {}

LazyDataFrame::LazyDataFrame(std::shared_ptr<LogicalNode> plan)
    : plan_(std::move(plan)) {}

std::shared_ptr<LogicalNode> LazyDataFrame::plan() const { return plan_; }


namespace {

template <typename NodeT>
LazyDataFrame extend(std::shared_ptr<LogicalNode> parent,
                     std::shared_ptr<NodeT> node) {
    if (!parent) {
        throw std::runtime_error(
            "LazyDataFrame: operation applied to an empty plan");
    }
    node->children = { parent };
    return LazyDataFrame(node);
}

} 


LazyDataFrame LazyDataFrame::select(const std::vector<std::string>& columns) const {
    auto node = std::make_shared<SelectNode>();
    node->columns.reserve(columns.size());
    for (const auto& n : columns) node->columns.emplace_back(col(n));
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::select(const std::vector<ExprBuilder>& exprs) const {
    auto node = std::make_shared<SelectNode>();
    node->columns = exprs;
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::select(std::initializer_list<const char*> columns) const {
    return select(std::vector<std::string>(columns.begin(), columns.end()));
}

LazyDataFrame LazyDataFrame::filter(const ExprBuilder& predicate) const {
    auto node       = std::make_shared<FilterNode>();
    node->predicate = predicate.expr();
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::with_column(const std::string& name,
                                         const ExprBuilder& expr) const {
    auto node  = std::make_shared<WithColNode>();
    node->name = name;
    node->expr = expr;
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::group_by(const std::vector<std::string>& keys) const {
    auto node  = std::make_shared<GroupByNode>();
    node->keys = keys;
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::aggregate(
    const std::map<std::string, ExprBuilder>& aggMap) const {
    auto node    = std::make_shared<AggNode>();
    node->aggMap = aggMap;
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& aggs) const {
    std::map<std::string, ExprBuilder> agg_map;

    for (const auto& [col_name, fn_name_raw] : aggs) {
        std::string fn_name = fn_name_raw;
        std::transform(
            fn_name.begin(), fn_name.end(), fn_name.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        ExprBuilder expr;
        if (fn_name == "sum") expr = col(col_name).sum();
        else if (fn_name == "mean") expr = col(col_name).mean();
        else if (fn_name == "count") expr = col(col_name).count();
        else if (fn_name == "min") expr = col(col_name).min();
        else if (fn_name == "max") expr = col(col_name).max();
        else {
            throw std::invalid_argument("aggregate: unsupported function '" + fn_name_raw + "'");
        }

        std::string out_name = col_name + "_" + fn_name;
        if (agg_map.find(out_name) != agg_map.end()) {
            int suffix = 2;
            std::string candidate;
            do {
                candidate = out_name + "_" + std::to_string(suffix++);
            } while (agg_map.find(candidate) != agg_map.end());
            out_name = candidate;
        }

        agg_map.emplace(std::move(out_name), std::move(expr));
    }

    return aggregate(agg_map);
}

LazyDataFrame LazyDataFrame::join(const LazyDataFrame& other,
                                  const std::vector<std::string>& on,
                                  const std::string& how) const {
    if (!other.plan_) throw std::runtime_error("join: right-hand plan is empty");
    auto node   = std::make_shared<JoinNode>();
    node->right = other.plan_;
    node->on    = on;
    node->how   = how;
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& columns,
                                  bool ascending) const {
    auto node        = std::make_shared<SortNode>();
    node->columns    = columns;
    node->ascending  = ascending;
    return extend(plan_, node);
}

LazyDataFrame LazyDataFrame::head(int64_t n) const {
    auto node = std::make_shared<LimitNode>();
    node->n   = n;
    return extend(plan_, node);
}


void LazyDataFrame::sink_csv(const std::string& path) const {
    collect().write_csv(path);
}

void LazyDataFrame::sink_parquet(const std::string& path) const {
    collect().write_parquet(path);
}


namespace {

EagerDataFrame executePlan(const std::shared_ptr<LogicalNode>& node);

const std::shared_ptr<LogicalNode>& requireChild(const LogicalNode& n,
                                                 const std::string& ctx) {
    if (n.children.empty() || !n.children[0]) {
        throw std::runtime_error(ctx + ": missing child plan");
    }
    return n.children[0];
}

EagerDataFrame runScan(const ScanNode& s) {
    if (s.path.empty()) {
        throw std::runtime_error("Scan: empty path");
    }
    auto df = s.isParquet ? read_parquet(s.path) : read_csv(s.path);

    if (!s.projected_columns.empty()) {
        std::vector<std::string> present;
        present.reserve(s.projected_columns.size());
        auto schema = df.table()->schema();
        for (const auto& c : s.projected_columns) {
            if (schema->GetFieldIndex(c) >= 0) present.push_back(c);
        }
        if (!present.empty()) df = df.select(present);
    }

    if (s.row_limit >= 0) df = df.head(s.row_limit);
    return df;
}

const std::vector<std::string>* groupKeysAbove(const LogicalNode& child) {
    if (auto g = dynamic_cast<const GroupByNode*>(&child)) return &g->keys;
    return nullptr;
}

EagerDataFrame executePlan(const std::shared_ptr<LogicalNode>& node) {
    if (!node) throw std::runtime_error("executePlan: null node");

    if (auto s = std::dynamic_pointer_cast<ScanNode>(node)) {
        return runScan(*s);
    }

    if (auto p = std::dynamic_pointer_cast<FilterNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "Filter"));
        return child_df.filter(ExprBuilder(p->predicate));
    }

    if (auto p = std::dynamic_pointer_cast<SelectNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "Select"));
        return child_df.select(p->columns);
    }

    if (auto p = std::dynamic_pointer_cast<WithColNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "WithColumn"));
        return child_df.with_column(p->name, p->expr);
    }

    if (auto p = std::dynamic_pointer_cast<GroupByNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "GroupBy"));
        return child_df.group_by(p->keys);
    }

    if (auto p = std::dynamic_pointer_cast<AggNode>(node)) {
        const auto& child = requireChild(*p, "Aggregate");

        if (const auto* keys = groupKeysAbove(*child)) {
            auto grand_child = executePlan(requireChild(*child, "Aggregate/GroupBy"));
            return grand_child.group_by(*keys).aggregate(p->aggMap);
        }

        auto child_df = executePlan(child);
        return child_df.aggregate(p->aggMap);
    }

    if (auto p = std::dynamic_pointer_cast<JoinNode>(node)) {
        auto left_df  = executePlan(requireChild(*p, "Join"));
        if (!p->right) throw std::runtime_error("Join: missing right plan");
        auto right_df = executePlan(p->right);
        return left_df.join(right_df, p->on, p->how);
    }

    if (auto p = std::dynamic_pointer_cast<SortNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "Sort"));
        return child_df.sort(p->columns, p->ascending);
    }

    if (auto p = std::dynamic_pointer_cast<LimitNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "Limit"));
        return child_df.head(p->n);
    }

    if (auto p = std::dynamic_pointer_cast<SinkNode>(node)) {
        auto child_df = executePlan(requireChild(*p, "Sink"));
        if (p->isParquet) child_df.write_parquet(p->path);
        else              child_df.write_csv(p->path);
        return child_df;
    }

    throw std::runtime_error("executePlan: unknown node type");
}

} 

EagerDataFrame LazyDataFrame::collect() const {
    if (!plan_) return EagerDataFrame();
    QueryOptimizer opt;
    auto optimized = opt.optimize(plan_);
    return executePlan(optimized);
}

EagerDataFrame LazyDataFrame::collect_raw() const {
    if (!plan_) return EagerDataFrame();
    return executePlan(plan_);
}


namespace {

struct TempDotFile {
    std::string path;
    bool        keep = false;

    explicit TempDotFile(std::string p) : path(std::move(p)) {}
    ~TempDotFile() {
        if (!keep && !path.empty()) std::remove(path.c_str());
    }

    TempDotFile(const TempDotFile&)            = delete;
    TempDotFile& operator=(const TempDotFile&) = delete;
};

} 

void LazyDataFrame::explain(const std::string& pngPath) const {
    QueryOptimizer opt;
    auto optimized = opt.optimize(plan_);
    const std::string dot = renderDotGraph(optimized);

    const auto ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    TempDotFile tmp("/tmp/plan_" + std::to_string(ts) + ".dot");

    {
        std::ofstream f(tmp.path);
        if (!f) {
            std::cerr << "Warning: explain() could not write " << tmp.path << "\n";
            tmp.keep = true; 
            return;
        }
        f << dot;
    }

    const std::string cmd =
        "dot -Tpng \"" + tmp.path + "\" -o \"" + pngPath + "\" 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        tmp.keep = true;
        std::cerr << "Warning: Graphviz not installed or `dot` failed. "
                     "DOT file written to " << tmp.path << "\n";
    }
}

} 
