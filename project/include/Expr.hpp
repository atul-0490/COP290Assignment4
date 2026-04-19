#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include <arrow/api.h>

#include "TypeUtils.hpp"

namespace dfl {

// ---------------------------------------------------------------------------
// Expression AST
// ---------------------------------------------------------------------------
// All expression nodes derive from `Expr`. Expressions are immutable value
// objects held via std::shared_ptr so they can be cheaply reused across
// lazy plans and eager evaluations.
// ---------------------------------------------------------------------------

struct Expr {
    virtual ~Expr() = default;
};

/// Reference to a column in the input schema by name.
struct ColExpr : Expr {
    std::string name;
};

/// Literal (constant) value. We store it as an arrow::Datum so that we can
/// reuse Arrow's compute kernels, plus the logical ColType for type-checking.
struct LitExpr : Expr {
    arrow::Datum value;
    ColType type;
};

/// Renames the result of `child` to `alias`.
struct AliasExpr : Expr {
    std::shared_ptr<Expr> child;
    std::string alias;
};

/// Binary arithmetic / comparison / boolean operator.
struct BinaryExpr : Expr {
    enum class Op {
        ADD, SUB, MUL, DIV, MOD,
        EQ, NEQ, LT, LE, GT, GE,
        AND, OR
    };

    std::shared_ptr<Expr> left;
    std::shared_ptr<Expr> right;
    Op op;
};

/// Unary numeric / boolean / null-check operator.
struct UnaryExpr : Expr {
    enum class Op {
        NEG,
        NOT,
        ABS,
        IS_NULL,
        IS_NOT_NULL
    };

    std::shared_ptr<Expr> child;
    Op op;
};

/// Aggregation function used within `aggregate()` expressions.
struct AggExpr : Expr {
    enum class Func { SUM, MEAN, COUNT, MIN, MAX };

    std::shared_ptr<Expr> child;
    Func func;
};

/// String function. `arg` is used only for CONTAINS / STARTS_WITH / ENDS_WITH.
struct StringExpr : Expr {
    enum class Func {
        LENGTH,
        CONTAINS,
        STARTS_WITH,
        ENDS_WITH,
        TO_LOWER,
        TO_UPPER
    };

    std::shared_ptr<Expr> child;
    Func func;
    std::string arg;
};

// ---------------------------------------------------------------------------
// Raw builder free-functions (produce std::shared_ptr<Expr> directly).
// The fluent `ExprBuilder` wrapper below is usually more ergonomic, but
// these are kept for internal use and for users who want raw nodes.
// ---------------------------------------------------------------------------

std::shared_ptr<Expr> makeCol(const std::string& name);

template <typename T>
std::shared_ptr<Expr> makeLit(T value);

// ---------------------------------------------------------------------------
// Fluent wrapper around a std::shared_ptr<Expr>.
// This is what user code sees, enabling chains like:
//
//     (col("age") > lit(30)) & (col("salary") < lit(100000.0))
//
// ---------------------------------------------------------------------------

class ExprBuilder {
public:
    ExprBuilder() = default;
    explicit ExprBuilder(std::shared_ptr<Expr> e);

    /// Unwrap the underlying Expr node.
    std::shared_ptr<Expr> expr() const;

    // ----- naming / unary -----
    ExprBuilder alias(const std::string& name) const;
    ExprBuilder abs() const;
    ExprBuilder is_null() const;
    ExprBuilder is_not_null() const;

    // ----- string functions -----
    ExprBuilder length() const;
    ExprBuilder contains(const std::string& s) const;
    ExprBuilder starts_with(const std::string& s) const;
    ExprBuilder ends_with(const std::string& s) const;
    ExprBuilder to_lower() const;
    ExprBuilder to_upper() const;

    // ----- aggregations -----
    ExprBuilder sum() const;
    ExprBuilder mean() const;
    ExprBuilder count() const;
    ExprBuilder min() const;
    ExprBuilder max() const;

    // ----- arithmetic -----
    ExprBuilder operator+(const ExprBuilder& rhs) const;
    ExprBuilder operator-(const ExprBuilder& rhs) const;
    ExprBuilder operator*(const ExprBuilder& rhs) const;
    ExprBuilder operator/(const ExprBuilder& rhs) const;
    ExprBuilder operator%(const ExprBuilder& rhs) const;

    // ----- comparisons -----
    ExprBuilder operator==(const ExprBuilder& rhs) const;
    ExprBuilder operator!=(const ExprBuilder& rhs) const;
    ExprBuilder operator< (const ExprBuilder& rhs) const;
    ExprBuilder operator<=(const ExprBuilder& rhs) const;
    ExprBuilder operator> (const ExprBuilder& rhs) const;
    ExprBuilder operator>=(const ExprBuilder& rhs) const;

    // ----- boolean -----
    ExprBuilder operator&(const ExprBuilder& rhs) const;
    ExprBuilder operator|(const ExprBuilder& rhs) const;
    ExprBuilder operator~() const;

private:
    std::shared_ptr<Expr> node_;
};

// ---------------------------------------------------------------------------
// Top-level builder entry points users call:
//
//     col("age")         → ExprBuilder wrapping a ColExpr
//     lit(42)            → ExprBuilder wrapping a LitExpr
// ---------------------------------------------------------------------------

ExprBuilder col(const std::string& name);

template <typename T>
ExprBuilder lit(T value);

// ---------------------------------------------------------------------------
// Template implementations — must live in the header so callers can
// instantiate with arbitrary T. Each simply stores the value inside a
// LitExpr and infers the logical ColType via constexpr type checks.
// ---------------------------------------------------------------------------

template <typename T>
std::shared_ptr<Expr> makeLit(T /*value*/) {
    // Stub: we only set the logical type in step 1. The concrete scalar
    // encoding inside `value` is wired up when the evaluator lands.
    auto node = std::make_shared<LitExpr>();
    if constexpr (std::is_same_v<T, int32_t>) {
        node->type = ColType::INT32;
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int>) {
        node->type = ColType::INT64;
    } else if constexpr (std::is_same_v<T, float>) {
        node->type = ColType::FLOAT32;
    } else if constexpr (std::is_same_v<T, double>) {
        node->type = ColType::FLOAT64;
    } else if constexpr (std::is_same_v<T, bool>) {
        node->type = ColType::BOOLEAN;
    } else {
        node->type = ColType::STRING;
    }
    return node;
}

template <typename T>
ExprBuilder lit(T value) {
    return ExprBuilder(makeLit<T>(value));
}

} // namespace dfl
