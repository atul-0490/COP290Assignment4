#include "LogicalPlan.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>



namespace dfl
{


namespace {

const char* binOpSym(BinaryExpr::Op op) {
    using Op = BinaryExpr::Op;
    switch (op) {
        case Op::ADD: return "+";
        case Op::SUB: return "-";
        case Op::MUL: return "*";
        case Op::DIV: return "/";
        case Op::MOD: return "%";
        case Op::EQ: return "==";
        case Op::NEQ: return "!=";
        case Op::LT: return "<";
        case Op::LE: return "<=";
        case Op::GT: return ">";
        case Op::GE: return ">=";
        case Op::AND: return "&";
        case Op::OR: return "|";
    }
    return "?";
}

const char* aggName(AggExpr::Func f) {
    using F = AggExpr::Func;
    switch (f) {
        case F::SUM: return "sum";
        case F::MEAN: return "mean";
        case F::COUNT: return "count";
        case F::MIN: return "min";
        case F::MAX: return "max";
    }
    return "?";
}

const char* strFuncName(StringExpr::Func f) {
    using F = StringExpr::Func;
    switch (f) {
        case F::LENGTH: return "length";
        case F::CONTAINS: return "contains";
        case F::STARTS_WITH: return "starts_with";
        case F::ENDS_WITH: return "ends_with";
        case F::TO_LOWER:  return "to_lower";
        case F::TO_UPPER: return "to_upper";
    }
    return "?";
}

std::string scalarToString(const arrow::Datum& d) {
    if (!d.is_scalar() || d.scalar() == nullptr) return "?";
    const auto& s = d.scalar();
    if (!s->is_valid) return "null";
    if (s->type && s->type->id() == arrow::Type::STRING) {
        return "\"" + s->ToString() + "\"";
    }
    return s->ToString();
}

} 

std::string exprToString(const std::shared_ptr<Expr>& e) {
    if (!e) return "?";

    if (auto c = std::dynamic_pointer_cast<ColExpr>(e))  return c->name;
    if (auto l = std::dynamic_pointer_cast<LitExpr>(e))  return scalarToString(l->value);

    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e)) {
        return exprToString(a->child) + " as " + a->alias;
    }

    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(e)) {
        return "(" + exprToString(b->left) + " " + binOpSym(b->op) + " " + exprToString(b->right) + ")";
    }

    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(e)) {
        using Op = UnaryExpr::Op;
        switch (u->op) {
            case Op::NEG: return "-" + exprToString(u->child);
            case Op::NOT:  return "~" + exprToString(u->child);
            case Op::ABS: return "abs(" + exprToString(u->child) + ")";
            case Op::IS_NULL: return exprToString(u->child) + ".is_null()";
            case Op::IS_NOT_NULL: return exprToString(u->child) + ".is_not_null()";
        }
    }

    if (auto g = std::dynamic_pointer_cast<AggExpr>(e)) {
        return std::string(aggName(g->func)) + "(" + exprToString(g->child) + ")";
    }

    if (auto s = std::dynamic_pointer_cast<StringExpr>(e)) {
        using F = StringExpr::Func;
        const std::string inner = exprToString(s->child);
        if (s->func == F::CONTAINS || s->func == F::STARTS_WITH || s->func == F::ENDS_WITH) {
            return std::string(strFuncName(s->func)) + "(" + inner +", \"" + s->arg + "\")";
        }
        return std::string(strFuncName(s->func)) + "(" + inner + ")";
    }

    return "?";
}


namespace {

std::string dotEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            default:   out += c; break;
        }
    }
    return out;
}

std::string joinStrings(const std::vector<std::string>& xs, const std::string& sep) {
    std::string out;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out += sep;
        out += xs[i];
    }
    return out;
}

std::pair<std::string, std::string> nodeAppearance(const LogicalNode& n) {
    if (auto p = dynamic_cast<const ScanNode*>(&n)) {
        return { "Scan\\n" + p->path + (p->isParquet ? " [parquet]" : " [csv]"), "lightblue" };
    }
    if (auto p = dynamic_cast<const FilterNode*>(&n)) {
        return { "Filter\\n" + exprToString(p->predicate), "lightyellow" };
    }
    if (auto p = dynamic_cast<const SelectNode*>(&n)) {
        std::vector<std::string> names;
        names.reserve(p->columns.size());
        for (const auto& eb : p->columns) names.push_back(exprToString(eb.expr()));
        return { "Select\\n[" + joinStrings(names, ", ") + "]", "lightgreen" };
    }
    if (auto p = dynamic_cast<const WithColNode*>(&n)) {
        return { "WithColumn\\n" + p->name + " = " + exprToString(p->expr.expr()),"lightgreen" };
    }
    if (auto p = dynamic_cast<const GroupByNode*>(&n)) {
        return { "GroupBy\\n[" + joinStrings(p->keys, ", ") + "]", "lightsalmon" };
    }
    if (auto p = dynamic_cast<const AggNode*>(&n)) {
        std::vector<std::string> parts;
        parts.reserve(p->aggMap.size());
        for (const auto& [name, eb] : p->aggMap) {
            parts.push_back(name + " = " + exprToString(eb.expr()));
        }
        return { "Aggregate\\n" + joinStrings(parts, "\\n"), "lightsalmon" };
    }
    if (auto p = dynamic_cast<const JoinNode*>(&n)) {
        return { "Join (" + p->how + ")\\non: [" + joinStrings(p->on, ", ") + "]", "plum" };
    }
    if (auto p = dynamic_cast<const SortNode*>(&n)) {
        return { "Sort\\n[" + joinStrings(p->columns, ", ") + "] " + (p->ascending ? "asc" : "desc"),"lightcyan" };
    }
    if (auto p = dynamic_cast<const LimitNode*>(&n)) {
        return { "Limit(" + std::to_string(p->n) + ")", "lightgray" };
    }
    if (auto p = dynamic_cast<const SinkNode*>(&n)) {
        return { "Sink\\n" + p->path + (p->isParquet ? " [parquet]" : " [csv]"), "orange" };
    }
    return { "Unknown", "white" };
}

class IdAssigner {
public:
    int idOf(const LogicalNode* n) {
        auto it = ids_.find(n);
        if (it != ids_.end()) return it->second;
        int id = next_++;
        ids_[n] = id;
        return id;
    }
    bool seen(const LogicalNode* n) const { return ids_.count(n) > 0; }

private:
    std::unordered_map<const LogicalNode*, int> ids_;
    int next_ = 0;
};

void walk(const std::shared_ptr<LogicalNode>& node, IdAssigner& ids,std::ostringstream& nodes_out,std::ostringstream& edges_out,std::unordered_map<const LogicalNode*, bool>& emitted) {
    if (!node) return;
    const auto* raw = node.get();
    if (emitted[raw]) return;
    emitted[raw] = true;

    const int id = ids.idOf(raw);
    auto [label, color] = nodeAppearance(*node);

    nodes_out << "  n" << id << " [label=\"" << dotEscape(label)
              << "\", style=filled, fillcolor=" << color
              << ", shape=box];\n";

    for (const auto& child : node->children) {
        if (!child) continue;
        walk(child, ids, nodes_out, edges_out, emitted);
        edges_out << "  n" << ids.idOf(child.get()) << " -> n" << id << ";\n";
    }

    if (auto jn = std::dynamic_pointer_cast<JoinNode>(node)) {
        if (jn->right) {
            walk(jn->right, ids, nodes_out, edges_out, emitted);
            edges_out << "  n" << ids.idOf(jn->right.get()) << " -> n" << id
                      << " [label=\"right\"];\n";
        }
    }
}

} 

std::string renderDotGraph(const std::shared_ptr<LogicalNode>& root) {
    std::ostringstream out;
    out << "digraph G {\n"
        << "  rankdir=TB;\n"
        << "  node [fontname=\"Helvetica\"];\n";

    if (!root) {
        out << "  empty [label=\"(empty plan)\", shape=box];\n}\n";
        return out.str();
    }

    IdAssigner ids;
    std::ostringstream nodes_out, edges_out;
    std::unordered_map<const LogicalNode*, bool> emitted;
    walk(root, ids, nodes_out, edges_out, emitted);

    out << nodes_out.str() << edges_out.str() << "}\n";
    return out.str();
}

} 
