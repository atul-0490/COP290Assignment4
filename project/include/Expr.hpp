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
// objects held via `std::shared_ptr` so they can be cheaply reused across
// lazy plans and eager evaluations.
// ---------------------------------------------------------------------------

/// @brief Abstract base of the expression AST.
///
/// Derive from this to add a new expression kind; the evaluator uses
/// `dynamic_pointer_cast` to dispatch on the concrete subclass. Every
/// concrete Expr is immutable after construction.
struct Expr {
    virtual ~Expr() = default;
};

/// @brief Reference to a column in the input schema by name.
///
/// Example: `col("age")` constructs a `ColExpr { name = "age" }`.
struct ColExpr : Expr {
    std::string name;
};

/// @brief Literal (constant) value.
///
/// Stored as an `arrow::Datum` so Arrow's compute kernels accept it
/// directly; the logical `ColType` is retained for static type-checking.
struct LitExpr : Expr {
    arrow::Datum value;
    ColType      type;
};

/// @brief Renames the result of `child` to `alias`.
///
/// Used by `col("x").alias("y")` and `select({ ... })` to choose output
/// column names.
struct AliasExpr : Expr {
    std::shared_ptr<Expr> child;
    std::string           alias;
};

/// @brief Binary arithmetic / comparison / boolean operator.
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

/// @brief Unary numeric / boolean / null-check operator.
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

/// @brief Aggregation function (sum / mean / count / min / max).
///
/// Only valid inside the map passed to `aggregate()`. Attempting to
/// evaluate an AggExpr during `filter()` or `select()` raises.
struct AggExpr : Expr {
    enum class Func { SUM, MEAN, COUNT, MIN, MAX };

    std::shared_ptr<Expr> child;
    Func func;
};

/// @brief String function. `arg` is used only for CONTAINS /
///        STARTS_WITH / ENDS_WITH.
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
    Func                  func;
    std::string           arg;
};

// ---------------------------------------------------------------------------
// Raw builder free-functions (produce std::shared_ptr<Expr> directly).
// The fluent `ExprBuilder` wrapper below is usually more ergonomic, but
// these are kept for internal use and for users who want raw nodes.
// ---------------------------------------------------------------------------

/// @brief Raw constructor for a `ColExpr` node.
/// @param name The referenced column's name.
/// @return     `std::shared_ptr<ColExpr>` upcast to Expr.
std::shared_ptr<Expr> makeCol(const std::string& name);

/// @brief Raw constructor for a `LitExpr` node carrying `value`.
///
/// Supports `int32_t`, `int64_t`, any other integer type (promoted to
/// int64), `float`, `double`, `bool`, and anything convertible to
/// `std::string` (string / const char* / char[]).
///
/// @tparam T     Any of the supported literal types.
/// @param value  The literal's value.
/// @return       `std::shared_ptr<LitExpr>` upcast to Expr.
template <typename T>
std::shared_ptr<Expr> makeLit(T value);

// ---------------------------------------------------------------------------
// Fluent wrapper around a std::shared_ptr<Expr>.
// This is what user code sees, enabling chains like:
//
//     (col("age") > lit(30)) & (col("salary") < lit(100000.0))
//
// ---------------------------------------------------------------------------

/// @brief Ergonomic, operator-overloading wrapper around `Expr`.
///
/// All member functions and operators return a new `ExprBuilder` holding
/// the appropriate `Expr` subclass. This enables terse, Pandas/Polars-
/// style expression composition:
///
/// @code
///   auto pred = (col("age") > lit(18)) & (col("income") < lit(50000.0));
///   df.filter(pred);
/// @endcode
class ExprBuilder {
public:
    ExprBuilder() = default;

    /// @brief Wrap an already-constructed `Expr` shared pointer.
    explicit ExprBuilder(std::shared_ptr<Expr> e);

    /// @brief Unwrap the underlying Expr node.
    /// @return The `shared_ptr<Expr>` this builder holds.
    std::shared_ptr<Expr> expr() const;

    // ----- naming / unary -----

    /// @brief Rename the expression's output column.
    /// @param name New output column name.
    /// @return     An `AliasExpr` builder.
    ExprBuilder alias(const std::string& name) const;

    /// @brief Absolute value of a numeric expression.
    /// @return A builder that computes `|x|` element-wise.
    /// @throws std::invalid_argument at evaluation time if the operand
    ///         is non-numeric.
    ExprBuilder abs() const;

    /// @brief Boolean column — `true` where the operand is null.
    ExprBuilder is_null() const;

    /// @brief Boolean column — `true` where the operand is non-null.
    ExprBuilder is_not_null() const;

    // ----- string functions -----

    /// @brief Unicode character length of a string column.
    /// @return An int32 column with the codepoint length of each string.
    /// @throws std::invalid_argument at eval time if operand is non-string.
    ExprBuilder length() const;

    /// @brief Match-count of the substring `s` inside each string.
    /// @param s The substring to search for.
    /// @return  An integer column giving the number of matches per row.
    ExprBuilder contains(const std::string& s) const;

    /// @brief Boolean "starts with" predicate on a string column.
    ExprBuilder starts_with(const std::string& s) const;

    /// @brief Boolean "ends with" predicate on a string column.
    ExprBuilder ends_with(const std::string& s) const;

    /// @brief Lowercase every element of a string column (UTF-8 aware).
    ExprBuilder to_lower() const;

    /// @brief Uppercase every element of a string column (UTF-8 aware).
    ExprBuilder to_upper() const;

    // ----- aggregations -----

    /// @brief Sum of values. Valid only inside `aggregate()`.
    ExprBuilder sum()   const;
    /// @brief Arithmetic mean. Valid only inside `aggregate()`.
    ExprBuilder mean()  const;
    /// @brief Count of non-null values. Valid only inside `aggregate()`.
    ExprBuilder count() const;
    /// @brief Minimum value. Valid only inside `aggregate()`.
    ExprBuilder min()   const;
    /// @brief Maximum value. Valid only inside `aggregate()`.
    ExprBuilder max()   const;

    // ----- arithmetic -----

    /// @brief Element-wise addition. @return `lhs + rhs`.
    ExprBuilder operator+(const ExprBuilder& rhs) const;
    /// @brief Element-wise subtraction. @return `lhs - rhs`.
    ExprBuilder operator-(const ExprBuilder& rhs) const;
    /// @brief Element-wise multiplication. @return `lhs * rhs`.
    ExprBuilder operator*(const ExprBuilder& rhs) const;
    /// @brief Element-wise division. @return `lhs / rhs`.
    ExprBuilder operator/(const ExprBuilder& rhs) const;
    /// @brief Element-wise modulo. @return `lhs % rhs`.
    ExprBuilder operator%(const ExprBuilder& rhs) const;

    // ----- comparisons -----

    /// @brief Element-wise equality. @return boolean column.
    ExprBuilder operator==(const ExprBuilder& rhs) const;
    /// @brief Element-wise inequality. @return boolean column.
    ExprBuilder operator!=(const ExprBuilder& rhs) const;
    /// @brief Element-wise less-than. @return boolean column.
    ExprBuilder operator< (const ExprBuilder& rhs) const;
    /// @brief Element-wise less-than-or-equal. @return boolean column.
    ExprBuilder operator<=(const ExprBuilder& rhs) const;
    /// @brief Element-wise greater-than. @return boolean column.
    ExprBuilder operator> (const ExprBuilder& rhs) const;
    /// @brief Element-wise greater-than-or-equal. @return boolean column.
    ExprBuilder operator>=(const ExprBuilder& rhs) const;

    // ----- boolean -----

    /// @brief Boolean AND (Kleene — null is propagated).
    ExprBuilder operator&(const ExprBuilder& rhs) const;

    /// @brief Boolean OR (Kleene — null is propagated).
    ExprBuilder operator|(const ExprBuilder& rhs) const;

    /// @brief Boolean NOT.
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

/// @brief Reference a column by name inside an expression.
///
/// The returned builder can be combined with operators and member
/// functions to build compound expressions.
///
/// @param name The column name to reference at evaluation time.
/// @return     An ExprBuilder wrapping a `ColExpr`.
///
/// @code
///   auto adults = df.filter(col("age") >= lit(18));
/// @endcode
ExprBuilder col(const std::string& name);

/// @brief Create a literal (constant) expression.
///
/// @tparam T    Any supported literal type (see `makeLit`).
/// @param value The literal value.
/// @return      An ExprBuilder wrapping a `LitExpr`.
template <typename T>
ExprBuilder lit(T value);

// ---------------------------------------------------------------------------
// Template implementations — must live in the header so callers can
// instantiate with arbitrary T. Each simply stores the value inside a
// LitExpr and infers the logical ColType via constexpr type checks.
// ---------------------------------------------------------------------------

template <typename T>
std::shared_ptr<Expr> makeLit(T value) {
    auto node = std::make_shared<LitExpr>();
    if constexpr (std::is_same_v<T, bool>) {
        node->type  = ColType::BOOLEAN;
        node->value = arrow::Datum(std::make_shared<arrow::BooleanScalar>(value));
    } else if constexpr (std::is_same_v<T, int32_t>) {
        node->type  = ColType::INT32;
        node->value = arrow::Datum(std::make_shared<arrow::Int32Scalar>(value));
    } else if constexpr (std::is_same_v<T, int64_t>) {
        node->type  = ColType::INT64;
        node->value = arrow::Datum(std::make_shared<arrow::Int64Scalar>(value));
    } else if constexpr (std::is_integral_v<T>) {
        // Fallback for `int`, `unsigned`, `long`, etc. — treat as int64.
        node->type  = ColType::INT64;
        node->value = arrow::Datum(std::make_shared<arrow::Int64Scalar>(
            static_cast<int64_t>(value)));
    } else if constexpr (std::is_same_v<T, float>) {
        node->type  = ColType::FLOAT32;
        node->value = arrow::Datum(std::make_shared<arrow::FloatScalar>(value));
    } else if constexpr (std::is_same_v<T, double>) {
        node->type  = ColType::FLOAT64;
        node->value = arrow::Datum(std::make_shared<arrow::DoubleScalar>(value));
    } else {
        // std::string, const char*, char[] — all route through string(...).
        node->type  = ColType::STRING;
        node->value = arrow::Datum(
            std::make_shared<arrow::StringScalar>(std::string(value)));
    }
    return node;
}

template <typename T>
ExprBuilder lit(T value) {
    return ExprBuilder(makeLit<T>(value));
}

} // namespace dfl
