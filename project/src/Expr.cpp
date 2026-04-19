#include "Expr.hpp"

namespace dfl {

// ---------------------------------------------------------------------------
// Raw builders
// ---------------------------------------------------------------------------

std::shared_ptr<Expr> makeCol(const std::string& name) {
    auto node  = std::make_shared<ColExpr>();
    node->name = name;
    return node;
}

// ---------------------------------------------------------------------------
// ExprBuilder — thin fluent wrapper around a shared_ptr<Expr>.
// Every chaining method constructs a new node so existing builders remain
// valid and reusable.
// ---------------------------------------------------------------------------

ExprBuilder::ExprBuilder(std::shared_ptr<Expr> e) : node_(std::move(e)) {}

std::shared_ptr<Expr> ExprBuilder::expr() const { return node_; }

// ----- naming / unary -----

ExprBuilder ExprBuilder::alias(const std::string& name) const {
    auto n   = std::make_shared<AliasExpr>();
    n->child = node_;
    n->alias = name;
    return ExprBuilder(n);
}

static ExprBuilder makeUnary(std::shared_ptr<Expr> child, UnaryExpr::Op op) {
    auto n   = std::make_shared<UnaryExpr>();
    n->child = std::move(child);
    n->op    = op;
    return ExprBuilder(n);
}

ExprBuilder ExprBuilder::abs()         const { return makeUnary(node_, UnaryExpr::Op::ABS); }
ExprBuilder ExprBuilder::is_null()     const { return makeUnary(node_, UnaryExpr::Op::IS_NULL); }
ExprBuilder ExprBuilder::is_not_null() const { return makeUnary(node_, UnaryExpr::Op::IS_NOT_NULL); }
ExprBuilder ExprBuilder::operator~()   const { return makeUnary(node_, UnaryExpr::Op::NOT); }

// ----- string functions -----

static ExprBuilder makeStr(std::shared_ptr<Expr> child,
                           StringExpr::Func f,
                           std::string arg = "") {
    auto n   = std::make_shared<StringExpr>();
    n->child = std::move(child);
    n->func  = f;
    n->arg   = std::move(arg);
    return ExprBuilder(n);
}

ExprBuilder ExprBuilder::length() const           { return makeStr(node_, StringExpr::Func::LENGTH); }
ExprBuilder ExprBuilder::contains(const std::string& s)     const { return makeStr(node_, StringExpr::Func::CONTAINS, s); }
ExprBuilder ExprBuilder::starts_with(const std::string& s)  const { return makeStr(node_, StringExpr::Func::STARTS_WITH, s); }
ExprBuilder ExprBuilder::ends_with(const std::string& s)    const { return makeStr(node_, StringExpr::Func::ENDS_WITH, s); }
ExprBuilder ExprBuilder::to_lower() const { return makeStr(node_, StringExpr::Func::TO_LOWER); }
ExprBuilder ExprBuilder::to_upper() const { return makeStr(node_, StringExpr::Func::TO_UPPER); }

// ----- aggregations -----

static ExprBuilder makeAgg(std::shared_ptr<Expr> child, AggExpr::Func f) {
    auto n   = std::make_shared<AggExpr>();
    n->child = std::move(child);
    n->func  = f;
    return ExprBuilder(n);
}

ExprBuilder ExprBuilder::sum()   const { return makeAgg(node_, AggExpr::Func::SUM); }
ExprBuilder ExprBuilder::mean()  const { return makeAgg(node_, AggExpr::Func::MEAN); }
ExprBuilder ExprBuilder::count() const { return makeAgg(node_, AggExpr::Func::COUNT); }
ExprBuilder ExprBuilder::min()   const { return makeAgg(node_, AggExpr::Func::MIN); }
ExprBuilder ExprBuilder::max()   const { return makeAgg(node_, AggExpr::Func::MAX); }

// ----- arithmetic / comparison / boolean -----

static ExprBuilder makeBin(std::shared_ptr<Expr> l,
                           std::shared_ptr<Expr> r,
                           BinaryExpr::Op op) {
    auto n   = std::make_shared<BinaryExpr>();
    n->left  = std::move(l);
    n->right = std::move(r);
    n->op    = op;
    return ExprBuilder(n);
}

ExprBuilder ExprBuilder::operator+(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::ADD); }
ExprBuilder ExprBuilder::operator-(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::SUB); }
ExprBuilder ExprBuilder::operator*(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::MUL); }
ExprBuilder ExprBuilder::operator/(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::DIV); }
ExprBuilder ExprBuilder::operator%(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::MOD); }

ExprBuilder ExprBuilder::operator==(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::EQ);  }
ExprBuilder ExprBuilder::operator!=(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::NEQ); }
ExprBuilder ExprBuilder::operator< (const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::LT);  }
ExprBuilder ExprBuilder::operator<=(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::LE);  }
ExprBuilder ExprBuilder::operator> (const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::GT);  }
ExprBuilder ExprBuilder::operator>=(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::GE);  }

ExprBuilder ExprBuilder::operator&(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::AND); }
ExprBuilder ExprBuilder::operator|(const ExprBuilder& rhs) const { return makeBin(node_, rhs.node_, BinaryExpr::Op::OR);  }

// ---------------------------------------------------------------------------
// Top-level builder
// ---------------------------------------------------------------------------

ExprBuilder col(const std::string& name) {
    return ExprBuilder(makeCol(name));
}

} // namespace dfl
