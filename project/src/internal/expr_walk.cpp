#include "internal/expr_walk.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <arrow/array/builder_binary.h>
#include <arrow/compute/api.h>

#include "internal/arx.hpp"



namespace dfl
{
namespace arx {
namespace {

std::shared_ptr<arrow::Array> mod_float_fallback_scan(const std::shared_ptr<arrow::Array>& L,const std::shared_ptr<arrow::Array>& R, int64_t nrows) {
    auto ty    = L->type();
    auto b     = pull(arrow::MakeBuilder(ty), "mod float: MakeBuilder");
    expect(b->Reserve(nrows), "mod float: Reserve");
    const bool is_f32 = ty->id() == arrow::Type::FLOAT;
    for (int64_t i = 0; i < nrows; ++i) {
        if (L->IsNull(i) || R->IsNull(i)) {
            expect(b->AppendNull(), "mod float: AppendNull");
            continue;
        }
        double lv;
        double rv;
        if (is_f32) {
            lv = std::static_pointer_cast<arrow::FloatArray>(L)->Value(i);
            rv = std::static_pointer_cast<arrow::FloatArray>(R)->Value(i);
        } else {
            lv = std::static_pointer_cast<arrow::DoubleArray>(L)->Value(i);
            rv = std::static_pointer_cast<arrow::DoubleArray>(R)->Value(i);
        }
        if (rv == 0.0) {
            expect(b->AppendNull(), "mod float: divide-by-zero null");
            continue;
        }
        if (is_f32) {
            float fv = static_cast<float>(std::fmod(lv, rv));
            expect(b->AppendScalar(arrow::FloatScalar(fv)), "mod float: AppendScalar");
        } else {
            expect(b->AppendScalar(arrow::DoubleScalar(std::fmod(lv, rv))),
                "mod float: AppendScalar");
        }
    }
    std::shared_ptr<arrow::Array> out;
    expect(b->Finish(&out), "mod float: Finish");
    return out;
}

std::shared_ptr<arrow::ChunkedArray> walk(const std::shared_ptr<arrow::Table>& table,
                                          const std::shared_ptr<Expr>& expr);

std::shared_ptr<arrow::ChunkedArray> col_ref(const std::shared_ptr<arrow::Table>& t,
                                             const ColExpr& e) {
    auto c = t->GetColumnByName(e.name);
    if (!c) throw std::runtime_error("column not found: " + e.name);
    return c;
}

std::shared_ptr<arrow::ChunkedArray> lit_broadcast(const std::shared_ptr<arrow::Table>& t,
                                                   const LitExpr& e) {
    if (!e.value.is_scalar()) throw std::runtime_error("LitExpr without scalar value");
    auto a = pull(arrow::MakeArrayFromScalar(*e.value.scalar(), t->num_rows()),
                  "LitExpr: MakeArrayFromScalar");
    return std::make_shared<arrow::ChunkedArray>(a);
}

std::shared_ptr<arrow::ChunkedArray> binary_op(const std::shared_ptr<arrow::Table>& t,
                                                const BinaryExpr& e) {
    auto lhs = walk(t, e.left);
    auto rhs = walk(t, e.right);
    typecheck_binary(kind_of(lhs), kind_of(rhs), e.op);

    if (e.op == BinaryExpr::Op::MOD) {
        std::shared_ptr<arrow::Array> Lflat;
        std::shared_ptr<arrow::Array> Rflat;
        if (lhs->num_chunks() == 0) {
            Lflat = pull(arrow::MakeArrayOfNull(lhs->type(), t->num_rows()), "mod: empty left fill");
        } else {
            Lflat = pull(arrow::Concatenate(lhs->chunks()), "mod: concat left chunks");
        }
        if (rhs->num_chunks() == 0) {
            Rflat = pull(arrow::MakeArrayOfNull(rhs->type(), t->num_rows()), "mod: empty right fill");
        } else {
            Rflat = pull(arrow::Concatenate(rhs->chunks()), "mod: concat right chunks");
        }

        const auto tid = Lflat->type_id();
        arrow::Datum Ld(Lflat);
        arrow::Datum Rd(Rflat);
        std::shared_ptr<arrow::Array> out_arr;

        if (tid == arrow::Type::INT32 || tid == arrow::Type::INT64) {
            auto q  = kernel("divide", {Ld, Rd});
            auto pr = kernel("multiply", {q, Rd});
            auto rem = kernel("subtract", {Ld, pr});
            std::shared_ptr<arrow::Array> rem_arr;
            if (rem.kind() == arrow::Datum::ARRAY) {
                rem_arr = rem.make_array();
            } else if (rem.kind() == arrow::Datum::CHUNKED_ARRAY) {
                rem_arr = pull(arrow::Concatenate(rem.chunked_array()->chunks()), "mod: concat rem");
            } else {
                throw std::runtime_error("mod: unexpected remainder datum kind");
            }

            std::shared_ptr<arrow::Scalar> z;
            switch (Rflat->type_id()) {
                case arrow::Type::INT32:
                    z = std::make_shared<arrow::Int32Scalar>(0);
                    break;
                case arrow::Type::INT64:
                    z = std::make_shared<arrow::Int64Scalar>(0);
                    break;
                default:
                    z = std::make_shared<arrow::Int64Scalar>(0);
                    break;
            }
            auto zmask = kernel("equal", {Rd, arrow::Datum(z)});
            auto nulls =
                pull(arrow::MakeArrayOfNull(rem_arr->type(), rem_arr->length()), "mod: MakeArrayOfNull");
            auto masked = kernel("if_else", {zmask, arrow::Datum(nulls), arrow::Datum(rem_arr)});
            if (masked.kind() == arrow::Datum::ARRAY) {
                out_arr = masked.make_array();
            } else if (masked.kind() == arrow::Datum::CHUNKED_ARRAY) {
                out_arr = pull(arrow::Concatenate(masked.chunked_array()->chunks()),
                                "mod: concat masked int");
            } else {
                throw std::runtime_error("mod: unexpected masked datum kind");
            }

            if (out_arr->type()->id() != tid) {
                arrow::compute::CastOptions copts;
                copts.to_type = Lflat->type();
                auto casted = arrow::compute::Cast(arrow::Datum(out_arr), copts);
                if (!casted.ok()) {
                    throw std::runtime_error("mod int cast: " + casted.status().ToString());
                }
                out_arr = casted.ValueOrDie().make_array();
            }
        } else if (tid == arrow::Type::FLOAT || tid == arrow::Type::DOUBLE) {
            auto fd_try = arrow::compute::CallFunction("floor_divide", {Ld, Rd});
            if (fd_try.ok()) {
                auto pr  = kernel("multiply", {fd_try.ValueOrDie(), Rd});
                auto rem = kernel("subtract", {Ld, pr});
                std::shared_ptr<arrow::Array> rem_arr;
                if (rem.kind() == arrow::Datum::ARRAY) {
                    rem_arr = rem.make_array();
                } else if (rem.kind() == arrow::Datum::CHUNKED_ARRAY) {
                    rem_arr = pull(arrow::Concatenate(rem.chunked_array()->chunks()),
                                    "mod: float rem concat");
                } else {
                    throw std::runtime_error("mod float: unexpected remainder datum kind");
                }
                std::shared_ptr<arrow::Scalar> z =
                    (tid == arrow::Type::FLOAT)
                        ? std::static_pointer_cast<arrow::Scalar>(
                              std::make_shared<arrow::FloatScalar>(0.0f))
                        : std::static_pointer_cast<arrow::Scalar>(
                              std::make_shared<arrow::DoubleScalar>(0.0));
                auto zmask = kernel("equal", {Rd, arrow::Datum(z)});
                auto nulls = pull(arrow::MakeArrayOfNull(rem_arr->type(), rem_arr->length()),
                                   "mod float: MakeArrayOfNull");
                auto masked =
                    kernel("if_else", {zmask, arrow::Datum(nulls), arrow::Datum(rem_arr)});
                if (masked.kind() == arrow::Datum::ARRAY) {
                    out_arr = masked.make_array();
                } else if (masked.kind() == arrow::Datum::CHUNKED_ARRAY) {
                    out_arr = pull(arrow::Concatenate(masked.chunked_array()->chunks()),
                                    "mod: concat masked float");
                } else {
                    throw std::runtime_error("mod float: unexpected masked datum kind");
                }
            } else {
                out_arr = mod_float_fallback_scan(Lflat, Rflat, t->num_rows());
            }
        } else {
            throw std::runtime_error("mod: unsupported operand type");
        }

        return std::make_shared<arrow::ChunkedArray>(out_arr);
    }

    const char* fn = nullptr;
    switch (e.op) {
        case BinaryExpr::Op::ADD:
            fn = "add";
            break;
        case BinaryExpr::Op::SUB:
            fn = "subtract";
            break;
        case BinaryExpr::Op::MUL:
            fn = "multiply";
            break;
        case BinaryExpr::Op::DIV:
            fn = "divide";
            break;
        case BinaryExpr::Op::EQ:
            fn = "equal";
            break;
        case BinaryExpr::Op::NEQ:
            fn = "not_equal";
            break;
        case BinaryExpr::Op::LT:
            fn = "less";
            break;
        case BinaryExpr::Op::LE:
            fn = "less_equal";
            break;
        case BinaryExpr::Op::GT:
            fn = "greater";
            break;
        case BinaryExpr::Op::GE:
            fn = "greater_equal";
            break;
        case BinaryExpr::Op::AND:
            fn = "and_kleene";
            break;
        case BinaryExpr::Op::OR:
            fn = "or_kleene";
            break;
        case BinaryExpr::Op::MOD:
            fn = nullptr;
            break;
    }

    auto raw = kernel(fn, {arrow::Datum(lhs), arrow::Datum(rhs)});
    return widen_to_chunked(raw, t->num_rows());
}

std::shared_ptr<arrow::ChunkedArray> unary_op(const std::shared_ptr<arrow::Table>& t,
                                               const UnaryExpr& e) {
    auto child = walk(t, e.child);
    const char* fn = nullptr;
    switch (e.op) {
        case UnaryExpr::Op::NEG:
            fn = "negate";
            break;
        case UnaryExpr::Op::NOT:
            if (kind_of(child) != ColType::BOOLEAN) {
                throw std::invalid_argument("NOT requires a boolean operand");
            }
            fn = "invert";
            break;
        case UnaryExpr::Op::ABS:
            if (!isNumeric(kind_of(child))) {
                throw std::invalid_argument("abs() requires a numeric operand");
            }
            fn = "abs";
            break;
        case UnaryExpr::Op::IS_NULL:
            fn = "is_null";
            break;
        case UnaryExpr::Op::IS_NOT_NULL:
            fn = "is_valid";
            break;
    }
    auto raw = kernel(fn, {arrow::Datum(child)});
    return widen_to_chunked(raw, t->num_rows());
}

std::shared_ptr<arrow::ChunkedArray> string_op(const std::shared_ptr<arrow::Table>& t,
                                                const StringExpr& e) {
    auto child = walk(t, e.child);
    if (kind_of(child) != ColType::STRING) {
        throw std::invalid_argument("string function requires a string operand");
    }
    using F = StringExpr::Func;
    switch (e.func) {
        case F::LENGTH:
            return widen_to_chunked(kernel("utf8_length", {arrow::Datum(child)}), t->num_rows());
        case F::CONTAINS: {
            arrow::compute::MatchSubstringOptions opt(e.arg);
            return widen_to_chunked(
                kernel("match_substring", {arrow::Datum(child)}, &opt), t->num_rows());
        }
        case F::STARTS_WITH: {
            arrow::compute::MatchSubstringOptions opt(e.arg);
            return widen_to_chunked(
                kernel("starts_with", {arrow::Datum(child)}, &opt), t->num_rows());
        }
        case F::ENDS_WITH: {
            arrow::compute::MatchSubstringOptions opt(e.arg);
            return widen_to_chunked(
                kernel("ends_with", {arrow::Datum(child)}, &opt), t->num_rows());
        }
        case F::TO_LOWER:
            return widen_to_chunked(kernel("utf8_lower", {arrow::Datum(child)}), t->num_rows());
        case F::TO_UPPER:
            return widen_to_chunked(kernel("utf8_upper", {arrow::Datum(child)}), t->num_rows());
    }
    throw std::runtime_error("unreachable: StringExpr::Func");
}

std::shared_ptr<arrow::ChunkedArray> agg_fold(const std::shared_ptr<arrow::Table>& t,
                                              const AggExpr& e) {
    auto child = walk(t, e.child);
    const char* fn = nullptr;
    switch (e.func) {
        case AggExpr::Func::SUM:
            fn = "sum";
            break;
        case AggExpr::Func::MEAN:
            fn = "mean";
            break;
        case AggExpr::Func::COUNT:
            fn = "count";
            break;
        case AggExpr::Func::MIN:
            fn = "min";
            break;
        case AggExpr::Func::MAX:
            fn = "max";
            break;
    }

    std::unique_ptr<arrow::compute::FunctionOptions> opts;
    if (e.func == AggExpr::Func::COUNT) {
        opts = std::make_unique<arrow::compute::CountOptions>();
    } else if (e.func == AggExpr::Func::MIN || e.func == AggExpr::Func::MAX ||
               e.func == AggExpr::Func::SUM || e.func == AggExpr::Func::MEAN) {
        opts = std::make_unique<arrow::compute::ScalarAggregateOptions>();
    }

    auto res = arrow::compute::CallFunction(fn, {arrow::Datum(child)}, opts.get());
    if (!res.ok()) {
        throw std::runtime_error(std::string("aggregate ") + fn + ": " + res.status().ToString());
    }
    auto scalar = res.ValueOrDie().scalar();
    auto one_row =
        pull(arrow::MakeArrayFromScalar(*scalar, 1), "agg: MakeArrayFromScalar");
    return std::make_shared<arrow::ChunkedArray>(one_row);
}

std::shared_ptr<arrow::ChunkedArray> walk(const std::shared_ptr<arrow::Table>& table,
                                          const std::shared_ptr<Expr>& expr) {
    if (!expr) throw std::runtime_error("evalExpr: null expression");
    if (!table) throw std::runtime_error("evalExpr: null table");

    if (auto c = std::dynamic_pointer_cast<ColExpr>(expr)) return col_ref(table, *c);
    if (auto l = std::dynamic_pointer_cast<LitExpr>(expr)) return lit_broadcast(table, *l);
    if (auto a = std::dynamic_pointer_cast<AliasExpr>(expr)) return walk(table, a->child);
    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(expr)) return binary_op(table, *b);
    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(expr)) return unary_op(table, *u);
    if (auto s = std::dynamic_pointer_cast<StringExpr>(expr)) return string_op(table, *s);
    if (auto g = std::dynamic_pointer_cast<AggExpr>(expr)) return agg_fold(table, *g);

    throw std::runtime_error("evalExpr: unknown expression node");
}

}  // namespace

std::shared_ptr<arrow::ChunkedArray> eval_on_table(const std::shared_ptr<arrow::Table>& table,
                                                   const std::shared_ptr<Expr>& expr) {
    return walk(table, expr);
}

}  // namespace arx
}  // namespace dfl
