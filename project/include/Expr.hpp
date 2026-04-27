#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include <arrow/api.h>

#include "TypeUtils.hpp"



namespace dfl
{

struct Expr {
    virtual ~Expr() = default;
};

struct ColExpr : Expr {
    std::string name;
};

struct LitExpr : Expr {
    arrow::Datum value;
    ColType      type;
};

struct AliasExpr : Expr {
    std::shared_ptr<Expr> child;
    std::string           alias;
};

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

struct AggExpr : Expr {
    enum class Func { SUM, MEAN, COUNT, MIN, MAX };

    std::shared_ptr<Expr> child;
    Func func;
};

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


std::shared_ptr<Expr> makeCol(const std::string& name);

template <typename T>
std::shared_ptr<Expr> makeLit(T value);


class ExprBuilder {
public:
    ExprBuilder() = default;

    explicit ExprBuilder(std::shared_ptr<Expr> e);

    std::shared_ptr<Expr> expr() const;


    ExprBuilder alias(const std::string& name) const;

    ExprBuilder abs() const;



    ExprBuilder is_null() const;

    ExprBuilder is_not_null() const;


    ExprBuilder length() const;

    ExprBuilder contains(const std::string& s) const;

    ExprBuilder starts_with(const std::string& s) const;


    ExprBuilder ends_with(const std::string& s) const;

    ExprBuilder to_lower() const;

    ExprBuilder to_upper() const;


    ExprBuilder sum()   const;
    ExprBuilder mean()  const;
    ExprBuilder count() const;
    ExprBuilder min()   const;
    ExprBuilder max()   const;


    ExprBuilder operator+(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator+(T rhs) const { return *this + ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator-(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator-(T rhs) const { return *this - ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator*(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator*(T rhs) const { return *this * ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator/(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator/(T rhs) const { return *this / ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator%(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator%(T rhs) const { return *this % ExprBuilder(makeLit<T>(rhs)); }


    ExprBuilder operator==(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator==(T rhs) const { return *this == ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator!=(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator!=(T rhs) const { return *this != ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator< (const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator< (T rhs) const { return *this < ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator<=(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator<=(T rhs) const { return *this <= ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator> (const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator> (T rhs) const { return *this > ExprBuilder(makeLit<T>(rhs)); }
    ExprBuilder operator>=(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator>=(T rhs) const { return *this >= ExprBuilder(makeLit<T>(rhs)); }


    ExprBuilder operator&(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator&(T rhs) const { return *this & ExprBuilder(makeLit<T>(rhs)); }

    ExprBuilder operator|(const ExprBuilder& rhs) const;
    template <typename T>
    ExprBuilder operator|(T rhs) const { return *this | ExprBuilder(makeLit<T>(rhs)); }

    ExprBuilder operator~() const;

private:
    std::shared_ptr<Expr> node_;
};


ExprBuilder col(const std::string& name);

template <typename T>
ExprBuilder lit(T value);


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

} 
