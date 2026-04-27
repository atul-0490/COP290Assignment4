#include "QueryOptimizer.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#ifdef DFL_HAVE_PARQUET
#include <parquet/arrow/reader.h>
#endif

#include "EagerDataFrame.hpp"  
#include "LogicalPlan.hpp"



namespace dfl
{


namespace {


bool exprEquals(const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b);

bool datumEquals(const arrow::Datum& a, const arrow::Datum& b) {
    if (!a.is_scalar() || !b.is_scalar()) return false;
    const auto& sa = a.scalar();
    const auto& sb = b.scalar();
    if (!sa || !sb) return sa.get() == sb.get();
    return sa->Equals(*sb);
}

bool exprEquals(const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) {
    if (a.get() == b.get()) return true;
    if (!a || !b) return false;

    if (auto ca = std::dynamic_pointer_cast<ColExpr>(a)) {
        auto cb = std::dynamic_pointer_cast<ColExpr>(b);
        return cb && ca->name == cb->name;
    }
    if (auto la = std::dynamic_pointer_cast<LitExpr>(a)) {
        auto lb = std::dynamic_pointer_cast<LitExpr>(b);
        return lb && datumEquals(la->value, lb->value);
    }
    if (auto xa = std::dynamic_pointer_cast<AliasExpr>(a)) {
        auto xb = std::dynamic_pointer_cast<AliasExpr>(b);
        return xb && xa->alias == xb->alias && exprEquals(xa->child, xb->child);
    }
    if (auto ba = std::dynamic_pointer_cast<BinaryExpr>(a)) {
        auto bb = std::dynamic_pointer_cast<BinaryExpr>(b);
        return bb && ba->op == bb->op &&
               exprEquals(ba->left,  bb->left) &&
               exprEquals(ba->right, bb->right);
    }
    if (auto ua = std::dynamic_pointer_cast<UnaryExpr>(a)) {
        auto ub = std::dynamic_pointer_cast<UnaryExpr>(b);
        return ub && ua->op == ub->op && exprEquals(ua->child, ub->child);
    }
    if (auto ga = std::dynamic_pointer_cast<AggExpr>(a)) {
        auto gb = std::dynamic_pointer_cast<AggExpr>(b);
        return gb && ga->func == gb->func && exprEquals(ga->child, gb->child);
    }
    if (auto sa = std::dynamic_pointer_cast<StringExpr>(a)) {
        auto sb = std::dynamic_pointer_cast<StringExpr>(b);
        return sb && sa->func == sb->func && sa->arg == sb->arg &&
               exprEquals(sa->child, sb->child);
    }
    return false;
}


void collectColRefs(const std::shared_ptr<Expr>& e, std::set<std::string>& out) {
    if (!e) return;
    if (auto c = std::dynamic_pointer_cast<ColExpr>(e)) { out.insert(c->name); return; }
    if (std::dynamic_pointer_cast<LitExpr>(e))          return;
    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e))  { collectColRefs(a->child, out); return; }
    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(e)) {
        collectColRefs(b->left,  out);
        collectColRefs(b->right, out);
        return;
    }
    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(e))  { collectColRefs(u->child, out); return; }
    if (auto g = std::dynamic_pointer_cast<AggExpr>(e))    { collectColRefs(g->child, out); return; }
    if (auto s = std::dynamic_pointer_cast<StringExpr>(e)) { collectColRefs(s->child, out); return; }
}

std::set<std::string> colRefs(const std::shared_ptr<Expr>& e) {
    std::set<std::string> s;
    collectColRefs(e, s);
    return s;
}

bool isConstant(const std::shared_ptr<Expr>& e) {
    if (!e) return false;
    if (std::dynamic_pointer_cast<ColExpr>(e)) return false;
    if (std::dynamic_pointer_cast<AggExpr>(e)) return false;
    if (std::dynamic_pointer_cast<LitExpr>(e)) return true;
    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e))  return isConstant(a->child);
    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(e)) return isConstant(b->left) && isConstant(b->right);
    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(e))  return isConstant(u->child);
    if (auto s = std::dynamic_pointer_cast<StringExpr>(e)) return isConstant(s->child);
    return false;
}


std::shared_ptr<arrow::Scalar> litScalar(const std::shared_ptr<Expr>& e) {
    auto l = std::dynamic_pointer_cast<LitExpr>(e);
    if (!l || !l->value.is_scalar()) return nullptr;
    return l->value.scalar();
}

bool litEquals(const std::shared_ptr<Expr>& e, int64_t v) {
    auto s = litScalar(e);
    if (!s || !s->is_valid) return false;
    switch (s->type->id()) {
        case arrow::Type::INT32:
            return std::static_pointer_cast<arrow::Int32Scalar>(s)->value == v;
        case arrow::Type::INT64:
            return std::static_pointer_cast<arrow::Int64Scalar>(s)->value == v;
        case arrow::Type::FLOAT:
            return std::static_pointer_cast<arrow::FloatScalar>(s)->value == static_cast<float>(v);
        case arrow::Type::DOUBLE:
            return std::static_pointer_cast<arrow::DoubleScalar>(s)->value == static_cast<double>(v);
        default:
            return false;
    }
}

bool litIsTrue(const std::shared_ptr<Expr>& e) {
    auto s = litScalar(e);
    if (!s || !s->is_valid || s->type->id() != arrow::Type::BOOL) return false;
    return std::static_pointer_cast<arrow::BooleanScalar>(s)->value;
}
bool litIsFalse(const std::shared_ptr<Expr>& e) {
    auto s = litScalar(e);
    if (!s || !s->is_valid || s->type->id() != arrow::Type::BOOL) return false;
    return !std::static_pointer_cast<arrow::BooleanScalar>(s)->value;
}

std::shared_ptr<Expr> makeLitFromScalar(const std::shared_ptr<arrow::Scalar>& s) {
    auto node = std::make_shared<LitExpr>();
    node->value = arrow::Datum(s);
    node->type  = arrowTypeToColType(s->type);
    return node;
}

std::shared_ptr<Expr> makeBoolLit(bool v) {
    auto node = std::make_shared<LitExpr>();
    node->type  = ColType::BOOLEAN;
    node->value = arrow::Datum(std::make_shared<arrow::BooleanScalar>(v));
    return node;
}

std::shared_ptr<Expr> makeIntLit(int32_t v) {
    auto node = std::make_shared<LitExpr>();
    node->type  = ColType::INT32;
    node->value = arrow::Datum(std::make_shared<arrow::Int32Scalar>(v));
    return node;
}


std::set<std::string> readCsvHeader(const std::string& path) {
    std::set<std::string> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    if (!std::getline(f, line)) return out;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.insert(line.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

std::set<std::string> readParquetColumns(const std::string& path) {
    std::set<std::string> out;
#ifdef DFL_HAVE_PARQUET
    auto infile_r = arrow::io::ReadableFile::Open(path);
    if (!infile_r.ok()) return out;
    auto reader_r = parquet::arrow::OpenFile(infile_r.ValueOrDie(), arrow::default_memory_pool());
    if (!reader_r.ok()) return out;
    std::shared_ptr<arrow::Schema> schema;
    if (!reader_r.ValueOrDie()->GetSchema(&schema).ok()) return out;
    for (const auto& f : schema->fields()) out.insert(f->name());
#else
    (void)path;
#endif
    return out;
}

std::set<std::string> scanOutputColumns(const ScanNode& s) {
    static std::mutex cache_mu;
    static std::unordered_map<std::string, std::set<std::string>> cache;

    const std::string key = (s.isParquet ? "pq:" : "csv:") + s.path;
    {
        std::lock_guard<std::mutex> lock(cache_mu);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    auto cols = s.isParquet ? readParquetColumns(s.path) : readCsvHeader(s.path);
    {
        std::lock_guard<std::mutex> lock(cache_mu);
        cache[key] = cols;
    }
    return cols;
}

std::optional<std::set<std::string>> outputColumns(
    const std::shared_ptr<LogicalNode>& node) {
    if (!node) return std::nullopt;

    if (auto s = std::dynamic_pointer_cast<ScanNode>(node)) {
        auto cols = scanOutputColumns(*s);
        if (cols.empty()) return std::nullopt;
        if (!s->projected_columns.empty()) {
            std::set<std::string> trimmed;
            for (const auto& c : s->projected_columns) {
                if (cols.count(c)) trimmed.insert(c);
            }
            return trimmed;
        }
        return cols;
    }

    if (auto p = std::dynamic_pointer_cast<SelectNode>(node)) {
        std::set<std::string> out;
        for (size_t i = 0; i < p->columns.size(); ++i) {
            auto e = p->columns[i].expr();
            if (auto a = std::dynamic_pointer_cast<AliasExpr>(e)) {
                out.insert(a->alias);
            } else if (auto c = std::dynamic_pointer_cast<ColExpr>(e)) {
                out.insert(c->name);
            } else {
                out.insert("expr_" + std::to_string(i));
            }
        }
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<WithColNode>(node)) {
        auto child = p->children.empty() ? std::optional<std::set<std::string>>{} : outputColumns(p->children[0]);
        if (!child) return std::nullopt;
        child->insert(p->name);
        return child;
    }

    if (auto p = std::dynamic_pointer_cast<AggNode>(node)) {
        std::set<std::string> out;
        if (!p->children.empty()) {
            if (auto g = std::dynamic_pointer_cast<GroupByNode>(p->children[0])) {out.insert(g->keys.begin(), g->keys.end());}
        }
        for (const auto& [k, _] : p->aggMap) out.insert(k);
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<JoinNode>(node)) {
        if (p->children.empty()) return std::nullopt;
        auto L = outputColumns(p->children[0]);
        auto R = outputColumns(p->right);
        if (!L || !R) return std::nullopt;
        std::set<std::string> onSet(p->on.begin(), p->on.end());
        std::set<std::string> out(L->begin(), L->end());
        for (const auto& c : *R) if (!onSet.count(c)) out.insert(c);
        return out;
    }

    if (node->children.empty()) return std::nullopt;
    return outputColumns(node->children[0]);
}


std::shared_ptr<LogicalNode> cloneNode(const std::shared_ptr<LogicalNode>& n) {
    if (auto p = std::dynamic_pointer_cast<ScanNode>(n)) {
        return std::make_shared<ScanNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<FilterNode>(n)) {
        return std::make_shared<FilterNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<SelectNode>(n)) {
        return std::make_shared<SelectNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<WithColNode>(n)) {
        return std::make_shared<WithColNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<GroupByNode>(n)) {
        return std::make_shared<GroupByNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<AggNode>(n)) {
        return std::make_shared<AggNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<JoinNode>(n)) {
        return std::make_shared<JoinNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<SortNode>(n)) {
        return std::make_shared<SortNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<LimitNode>(n)) {
        return std::make_shared<LimitNode>(*p);
    }
    if (auto p = std::dynamic_pointer_cast<SinkNode>(n)) {
        return std::make_shared<SinkNode>(*p);
    }
    return n;
}


bool planEquals(const std::shared_ptr<LogicalNode>& a, const std::shared_ptr<LogicalNode>& b);

bool childrenEqual(const std::shared_ptr<LogicalNode>& a, const std::shared_ptr<LogicalNode>& b) {
    if (a->children.size() != b->children.size()) return false;
    for (size_t i = 0; i < a->children.size(); ++i) {
        if (!planEquals(a->children[i], b->children[i])) return false;
    }
    return true;
}

bool planEquals(const std::shared_ptr<LogicalNode>& a,
                const std::shared_ptr<LogicalNode>& b) {
    if (a.get() == b.get()) return true;
    if (!a || !b) return false;

    if (auto pa = std::dynamic_pointer_cast<ScanNode>(a)) {
        auto pb = std::dynamic_pointer_cast<ScanNode>(b);
        if (!pb) return false;
        return pa->path == pb->path && pa->isParquet == pb->isParquet &&pa->projected_columns == pb->projected_columns &&pa->row_limit == pb->row_limit;
    }
    if (auto pa = std::dynamic_pointer_cast<FilterNode>(a)) {
        auto pb = std::dynamic_pointer_cast<FilterNode>(b);
        return pb && exprEquals(pa->predicate, pb->predicate) && childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<SelectNode>(a)) {
        auto pb = std::dynamic_pointer_cast<SelectNode>(b);
        if (!pb || pa->columns.size() != pb->columns.size()) return false;
        for (size_t i = 0; i < pa->columns.size(); ++i) {
            if (!exprEquals(pa->columns[i].expr(), pb->columns[i].expr())) return false;
        }
        return childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<WithColNode>(a)) {
        auto pb = std::dynamic_pointer_cast<WithColNode>(b);
        return pb && pa->name == pb->name &&exprEquals(pa->expr.expr(), pb->expr.expr()) &&childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<GroupByNode>(a)) {
        auto pb = std::dynamic_pointer_cast<GroupByNode>(b);
        return pb && pa->keys == pb->keys && childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<AggNode>(a)) {
        auto pb = std::dynamic_pointer_cast<AggNode>(b);
        if (!pb || pa->aggMap.size() != pb->aggMap.size()) return false;
        auto it1 = pa->aggMap.begin();
        auto it2 = pb->aggMap.begin();
        while (it1 != pa->aggMap.end()) {
            if (it1->first != it2->first || !exprEquals(it1->second.expr(), it2->second.expr())) return false;
            ++it1; 
            ++it2;
        }
        return childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<JoinNode>(a)) {
        auto pb = std::dynamic_pointer_cast<JoinNode>(b);
        return pb && pa->on == pb->on && pa->how == pb->how && planEquals(pa->right, pb->right) && childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<SortNode>(a)) {
        auto pb = std::dynamic_pointer_cast<SortNode>(b);
        return pb && pa->columns == pb->columns && pa->ascending == pb->ascending &&childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<LimitNode>(a)) {
        auto pb = std::dynamic_pointer_cast<LimitNode>(b);
        return pb && pa->n == pb->n && childrenEqual(a, b);
    }
    if (auto pa = std::dynamic_pointer_cast<SinkNode>(a)) {
        auto pb = std::dynamic_pointer_cast<SinkNode>(b);
        return pb && pa->path == pb->path && pa->isParquet == pb->isParquet &&childrenEqual(a, b);
    }
    return false;
}


std::shared_ptr<Expr> foldExpr(const std::shared_ptr<Expr>& e);

std::shared_ptr<Expr> evalToLit(const std::shared_ptr<Expr>& e) {
    try {
        auto empty_schema = arrow::schema({arrow::field("_", arrow::int32())});
        arrow::Int32Builder b;
        (void)b.Append(0);
        std::shared_ptr<arrow::Array> arr;
        (void)b.Finish(&arr);
        auto table = arrow::Table::Make(empty_schema,{ std::make_shared<arrow::ChunkedArray>(arr) });
        EagerDataFrame df(table);
        auto chunked = df.evalExpr(e);
        auto scalar  = chunked->GetScalar(0).ValueOrDie();
        return makeLitFromScalar(scalar);
    } catch (...) {
        return e;
    }
}

std::shared_ptr<Expr> foldExpr(const std::shared_ptr<Expr>& e) {
    if (!e) return e;

    if (std::dynamic_pointer_cast<ColExpr>(e)) return e;
    if (std::dynamic_pointer_cast<LitExpr>(e)) return e;

    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e)) {
        auto child = foldExpr(a->child);
        if (child == a->child) return e;
        auto out = std::make_shared<AliasExpr>(*a);
        out->child = child;
        return out;
    }
    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(e)) {
        auto L = foldExpr(b->left);
        auto R = foldExpr(b->right);
        if (isConstant(L) && isConstant(R)) {
            auto folded = evalToLit(e);
            if (std::dynamic_pointer_cast<LitExpr>(folded)) return folded;
        }
        if (L == b->left && R == b->right) return e;
        auto out = std::make_shared<BinaryExpr>(*b);
        out->left = L; out->right = R;
        return out;
    }
    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(e)) {
        auto C = foldExpr(u->child);
        if (isConstant(C)) {
            auto folded = evalToLit(e);
            if (std::dynamic_pointer_cast<LitExpr>(folded)) return folded;
        }
        if (C == u->child) return e;
        auto out = std::make_shared<UnaryExpr>(*u);
        out->child = C;
        return out;
    }
    if (auto g = std::dynamic_pointer_cast<AggExpr>(e)) {
        auto C = foldExpr(g->child);
        if (C == g->child) return e;
        auto out = std::make_shared<AggExpr>(*g);
        out->child = C;
        return out;
    }
    if (auto s = std::dynamic_pointer_cast<StringExpr>(e)) {
        auto C = foldExpr(s->child);
        if (isConstant(C)) {
            auto folded = evalToLit(e);
            if (std::dynamic_pointer_cast<LitExpr>(folded)) return folded;
        }
        if (C == s->child) return e;
        auto out = std::make_shared<StringExpr>(*s);
        out->child = C;
        return out;
    }
    return e;
}


std::shared_ptr<Expr> simplifyExpr(const std::shared_ptr<Expr>& e) {
    if (!e) return e;

    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e)) {
        auto child = simplifyExpr(a->child);
        if (child == a->child) return e;
        auto out = std::make_shared<AliasExpr>(*a);
        out->child = child;
        return out;
    }

    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(e)) {
        auto L = simplifyExpr(b->left);
        auto R = simplifyExpr(b->right);
        using Op = BinaryExpr::Op;

        if (b->op == Op::MUL) {
            if (litEquals(R, 1)) return L;
            if (litEquals(L, 1)) return R;
            if (litEquals(R, 0)) return R;           
            if (litEquals(L, 0)) return L;
        }
        if (b->op == Op::ADD) {
            if (litEquals(R, 0)) return L;
            if (litEquals(L, 0)) return R;
        }
        if (b->op == Op::SUB) {
            if (litEquals(R, 0)) return L;
            if (std::dynamic_pointer_cast<LitExpr>(L) &&
                std::dynamic_pointer_cast<LitExpr>(R) && exprEquals(L, R)) {
                return makeIntLit(0);                
            }
        }
        if (b->op == Op::DIV && litEquals(R, 1)) return L;

        if (b->op == Op::AND) {
            if (litIsTrue(R))  return L;
            if (litIsTrue(L))  return R;
            if (litIsFalse(R)) return R;             
            if (litIsFalse(L)) return L;
        }
        if (b->op == Op::OR) {
            if (litIsFalse(R)) return L;
            if (litIsFalse(L)) return R;
            if (litIsTrue(R))  return R;             
            if (litIsTrue(L))  return L;
        }

        if (b->op == Op::EQ && std::dynamic_pointer_cast<LitExpr>(L) &&
            std::dynamic_pointer_cast<LitExpr>(R) && exprEquals(L, R)) {
            return makeBoolLit(true);
        }

        if (L == b->left && R == b->right) return e;
        auto out = std::make_shared<BinaryExpr>(*b);
        out->left = L; out->right = R;
        return out;
    }

    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(e)) {
        auto C = simplifyExpr(u->child);
        if (u->op == UnaryExpr::Op::NOT) {
            if (auto inner = std::dynamic_pointer_cast<UnaryExpr>(C)) {
                if (inner->op == UnaryExpr::Op::NOT) return inner->child;
            }
        }
        if (C == u->child) return e;
        auto out = std::make_shared<UnaryExpr>(*u);
        out->child = C;
        return out;
    }

    if (auto g = std::dynamic_pointer_cast<AggExpr>(e)) {
        auto C = simplifyExpr(g->child);
        if (C == g->child) return e;
        auto out = std::make_shared<AggExpr>(*g);
        out->child = C;
        return out;
    }

    if (auto s = std::dynamic_pointer_cast<StringExpr>(e)) {
        auto C = simplifyExpr(s->child);
        if (C == s->child) return e;
        auto out = std::make_shared<StringExpr>(*s);
        out->child = C;
        return out;
    }
    return e;
}


using ExprRewriter = std::shared_ptr<Expr> (*)(const std::shared_ptr<Expr>&);

std::shared_ptr<LogicalNode> rewriteNodeExprs(
    const std::shared_ptr<LogicalNode>& node, ExprRewriter fn) {
    if (!node) return node;
    if (auto p = std::dynamic_pointer_cast<FilterNode>(node)) {
        auto newPred = fn(p->predicate);
        if (newPred == p->predicate) return node;
        auto out = std::make_shared<FilterNode>(*p);
        out->predicate = newPred;
        return out;
    }
    if (auto p = std::dynamic_pointer_cast<SelectNode>(node)) {
        bool changed = false;
        std::vector<ExprBuilder> newCols;
        newCols.reserve(p->columns.size());
        for (const auto& eb : p->columns) {
            auto rewritten = fn(eb.expr());
            if (rewritten != eb.expr()) changed = true;
            newCols.emplace_back(rewritten);
        }
        if (!changed) return node;
        auto out = std::make_shared<SelectNode>(*p);
        out->columns = std::move(newCols);
        return out;
    }
    if (auto p = std::dynamic_pointer_cast<WithColNode>(node)) {
        auto rewritten = fn(p->expr.expr());
        if (rewritten == p->expr.expr()) return node;
        auto out = std::make_shared<WithColNode>(*p);
        out->expr = ExprBuilder(rewritten);
        return out;
    }
    if (auto p = std::dynamic_pointer_cast<AggNode>(node)) {
        bool changed = false;
        std::map<std::string, ExprBuilder> newMap;
        for (const auto& [name, eb] : p->aggMap) {
            auto rewritten = fn(eb.expr());
            if (rewritten != eb.expr()) changed = true;
            newMap.emplace(name, ExprBuilder(rewritten));
        }
        if (!changed) return node;
        auto out = std::make_shared<AggNode>(*p);
        out->aggMap = std::move(newMap);
        return out;
    }
    return node;
}

using NodeRewriter = std::function<std::shared_ptr<LogicalNode>(
    const std::shared_ptr<LogicalNode>&)>;

std::shared_ptr<LogicalNode> mapChildren(
    const std::shared_ptr<LogicalNode>& node,
    const NodeRewriter& fn) {
    if (!node) return node;

    std::vector<std::shared_ptr<LogicalNode>> newChildren;
    newChildren.reserve(node->children.size());
    bool changed = false;
    for (const auto& c : node->children) {
        auto nc = fn(c);
        if (nc != c) changed = true;
        newChildren.push_back(nc);
    }

    std::shared_ptr<LogicalNode> newRight;
    if (auto jn = std::dynamic_pointer_cast<JoinNode>(node)) {
        newRight = fn(jn->right);
        if (newRight != jn->right) changed = true;
    }

    if (!changed) return node;
    auto out = cloneNode(node);
    out->children = std::move(newChildren);
    if (auto jn = std::dynamic_pointer_cast<JoinNode>(out)) {
        jn->right = newRight;
    }
    return out;
}


std::shared_ptr<LogicalNode> predicatePushdownRec(
    const std::shared_ptr<LogicalNode>& node);

std::shared_ptr<LogicalNode> predicatePushdownRec(
    const std::shared_ptr<LogicalNode>& node) {
    if (!node) return node;

    auto rewritten = mapChildren(node, predicatePushdownRec);

    auto filter = std::dynamic_pointer_cast<FilterNode>(rewritten);
    if (!filter) return rewritten;
    if (filter->children.empty()) return rewritten;

    auto child = filter->children[0];
    auto refs  = colRefs(filter->predicate);

    if (auto jn = std::dynamic_pointer_cast<JoinNode>(child)) {
        auto left_cols  = outputColumns(jn->children[0]);
        auto right_cols = outputColumns(jn->right);
        if (left_cols && right_cols) {
            bool all_left  = true, all_right = true;
            for (const auto& c : refs) {
                if (!left_cols->count(c))  all_left  = false;
                if (!right_cols->count(c)) all_right = false;
            }
            auto pushFilter = [&](const std::shared_ptr<LogicalNode>& side) {
                auto f = std::make_shared<FilterNode>();
                f->predicate = filter->predicate;
                f->children  = { side };
                return std::shared_ptr<LogicalNode>(f);
            };
            if (all_left && !refs.empty()) {
                auto newJn = std::make_shared<JoinNode>(*jn);
                newJn->children = { pushFilter(jn->children[0]) };
                return newJn;
            }
            if (all_right && !refs.empty()) {
                auto newJn = std::make_shared<JoinNode>(*jn);
                newJn->right = pushFilter(jn->right);
                return newJn;
            }
        }
    }

    if (auto gb = std::dynamic_pointer_cast<GroupByNode>(child)) {
        bool all_keys = !refs.empty();
        std::set<std::string> keySet(gb->keys.begin(), gb->keys.end());
        for (const auto& c : refs) if (!keySet.count(c)) { all_keys = false; break; }
        if (all_keys && !gb->children.empty()) {
            auto f = std::make_shared<FilterNode>();
            f->predicate = filter->predicate;
            f->children  = { gb->children[0] };
            auto newGb = std::make_shared<GroupByNode>(*gb);
            newGb->children = { f };
            return newGb;
        }
    }

    return rewritten;
}


std::shared_ptr<LogicalNode> projectionPushdownRec( const std::shared_ptr<LogicalNode>& node, const std::optional<std::set<std::string>>& required);

std::optional<std::set<std::string>> augment(
    const std::optional<std::set<std::string>>& req,
    const std::set<std::string>& extras) {
    if (!req) return std::nullopt;
    auto out = *req;
    out.insert(extras.begin(), extras.end());
    return out;
}

std::shared_ptr<LogicalNode> projectionPushdownRec(
    const std::shared_ptr<LogicalNode>& node,
    const std::optional<std::set<std::string>>& required) {
    if (!node) return node;

    if (auto s = std::dynamic_pointer_cast<ScanNode>(node)) {
        // NOTE: we can only push projections into scans for files we can
        // inspect the schema of ahead of time. For in-memory frames we skip.
        if (!required) return node;

        auto file_cols = scanOutputColumns(*s);
        if (file_cols.empty()) return node;

        std::vector<std::string> wanted;
        wanted.reserve(required->size());
        for (const auto& c : *required) {
            if (file_cols.count(c)) wanted.push_back(c);
        }
        std::sort(wanted.begin(), wanted.end());

        if (wanted.empty()) return node;
        if (wanted == s->projected_columns) return node;

        auto out = std::make_shared<ScanNode>(*s);
        out->projected_columns = std::move(wanted);
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<SelectNode>(node)) {
        std::set<std::string> needs;
        for (const auto& eb : p->columns) collectColRefs(eb.expr(), needs);
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], needs);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<SelectNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<FilterNode>(node)) {
        auto pred_refs = colRefs(p->predicate);
        auto child_req = augment(required, pred_refs);
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], child_req);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<FilterNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<WithColNode>(node)) {
        std::set<std::string> expr_refs;
        collectColRefs(p->expr.expr(), expr_refs);

        std::optional<std::set<std::string>> child_req;
        if (required) {
            child_req = *required;
            child_req->erase(p->name);           
            child_req->insert(expr_refs.begin(), expr_refs.end());
        }
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], child_req);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<WithColNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<SortNode>(node)) {
        auto child_req = required;
        if (child_req) { child_req->insert(p->columns.begin(), p->columns.end());}
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], child_req);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<SortNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<LimitNode>(node)) {
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], required);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<LimitNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<GroupByNode>(node)) {
        auto child_req = required;
        if (child_req) { child_req->insert(p->keys.begin(), p->keys.end()); }
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], child_req);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<GroupByNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<AggNode>(node)) {
        std::set<std::string> needs;
        for (const auto& [_, eb] : p->aggMap) collectColRefs(eb.expr(), needs);
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], needs);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<AggNode>(*p);
        out->children = { newChild };
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<JoinNode>(node)) {
        auto left_cols  = outputColumns(p->children.empty() ? nullptr : p->children[0]);
        auto right_cols = outputColumns(p->right);
        std::set<std::string> onKeys(p->on.begin(), p->on.end());

        std::optional<std::set<std::string>> leftReq, rightReq;
        if (required && left_cols) {
            leftReq = std::set<std::string>{};
            for (const auto& c : *required)  if (left_cols->count(c))  leftReq->insert(c);
            leftReq->insert(onKeys.begin(), onKeys.end());
        }
        if (required && right_cols) {
            rightReq = std::set<std::string>{};
            for (const auto& c : *required) if (right_cols->count(c)) rightReq->insert(c);
            rightReq->insert(onKeys.begin(), onKeys.end());
        }

        auto newLeft  = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], leftReq);
        auto newRight = projectionPushdownRec(p->right, rightReq);

        if ((p->children.empty() || newLeft == p->children.front()) && newRight == p->right) return node;
        auto out = std::make_shared<JoinNode>(*p);
        if (!p->children.empty()) out->children = { newLeft };
        out->right = newRight;
        return out;
    }

    if (auto p = std::dynamic_pointer_cast<SinkNode>(node)) {
        auto newChild = p->children.empty() ? std::shared_ptr<LogicalNode>{} : projectionPushdownRec(p->children[0], required);
        if (!newChild || newChild == p->children.front()) return node;
        auto out = std::make_shared<SinkNode>(*p);
        out->children = { newChild };
        return out;
    }

    return node;
}


std::shared_ptr<LogicalNode> limitPushdownRec(
    const std::shared_ptr<LogicalNode>& node) {
    if (!node) return node;
    auto rewritten = mapChildren(node, limitPushdownRec);

    auto lim = std::dynamic_pointer_cast<LimitNode>(rewritten);
    if (!lim || lim->children.empty()) return rewritten;

    auto child = lim->children[0];

    if (auto sel = std::dynamic_pointer_cast<SelectNode>(child)) {
        if (sel->children.empty()) return rewritten;
        auto newLim = std::make_shared<LimitNode>();
        newLim->n = lim->n;
        newLim->children = { sel->children[0] };
        auto newSel = std::make_shared<SelectNode>(*sel);
        newSel->children = { newLim };
        return newSel;
    }

    if (auto wc = std::dynamic_pointer_cast<WithColNode>(child)) {
        if (wc->children.empty()) return rewritten;
        auto newLim = std::make_shared<LimitNode>();
        newLim->n = lim->n;
        newLim->children = { wc->children[0] };
        auto newWc = std::make_shared<WithColNode>(*wc);
        newWc->children = { newLim };
        return newWc;
    }

    if (auto s = std::dynamic_pointer_cast<ScanNode>(child)) {
        int64_t new_limit = lim->n;
        if (s->row_limit >= 0) new_limit = std::min<int64_t>(s->row_limit, new_limit);
        if (s->row_limit == new_limit) {
            return child;
        }
        auto out = std::make_shared<ScanNode>(*s);
        out->row_limit = new_limit;
        return out;
    }

    return rewritten;
}

} 


std::shared_ptr<LogicalNode> QueryOptimizer::predicatePushdown(
    const std::shared_ptr<LogicalNode>& node) const {
    return predicatePushdownRec(node);
}

std::shared_ptr<LogicalNode> QueryOptimizer::projectionPushdown(
    const std::shared_ptr<LogicalNode>& node) const {
    return projectionPushdownRec(node, std::nullopt);
}

std::shared_ptr<LogicalNode> QueryOptimizer::constantFolding(
    const std::shared_ptr<LogicalNode>& node) const {
    NodeRewriter rewriter = [&](const std::shared_ptr<LogicalNode>& n) -> std::shared_ptr<LogicalNode> {
        auto kids = mapChildren(n, rewriter);
        return rewriteNodeExprs(kids, &foldExpr);
    };
    return rewriter(node);
}

std::shared_ptr<LogicalNode> QueryOptimizer::expressionSimplification(
    const std::shared_ptr<LogicalNode>& node) const {
    NodeRewriter rewriter = [&](const std::shared_ptr<LogicalNode>& n) -> std::shared_ptr<LogicalNode> {
        auto kids = mapChildren(n, rewriter);
        return rewriteNodeExprs(kids, &simplifyExpr);
    };
    return rewriter(node);
}

std::shared_ptr<LogicalNode> QueryOptimizer::limitPushdown(
    const std::shared_ptr<LogicalNode>& node) const {
    return limitPushdownRec(node);
}

std::shared_ptr<LogicalNode> QueryOptimizer::optimize(
    const std::shared_ptr<LogicalNode>& plan) const {
    if (!plan) return plan;
    auto current = plan;
    for (int iter = 0; iter < 10; ++iter) {
        auto next = current;
        next = constantFolding(next);
        next = expressionSimplification(next);
        next = predicatePushdown(next);
        next = projectionPushdown(next);
        next = limitPushdown(next);
        if (planEquals(next, current)) { current = next; break; }
        current = next;
    }
    return current;
}

} 
