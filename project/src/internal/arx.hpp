#pragma once

#include <arrow/compute/api.h>
#include <arrow/compute/initialize.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Expr.hpp"
#include "TypeUtils.hpp"



namespace dfl
{
namespace arx {

inline void prime_kernels() {
    static const bool once = [] {
        auto st = arrow::compute::Initialize();
        if (!st.ok()) {
            throw std::runtime_error(std::string("arrow::compute::Initialize: ") + st.ToString());
        }
        return true;
    }();
    (void)once;
}

inline void expect(const arrow::Status& st, const std::string& ctx) {
    if (!st.ok()) {
        throw std::runtime_error(ctx + ": " + st.ToString());
    }
}

template <typename T>
inline T pull(arrow::Result<T>&& r, const std::string& ctx) {
    if (!r.ok()) {
        throw std::runtime_error(ctx + ": " + r.status().ToString());
    }
    return std::move(r).ValueOrDie();
}

inline arrow::Datum kernel(const std::string& fn,
                           std::vector<arrow::Datum> args,
                           const arrow::compute::FunctionOptions* opts = nullptr) {
    auto out = arrow::compute::CallFunction(fn, std::move(args), opts);
    if (!out.ok()) {
        throw std::runtime_error("compute::" + fn + ": " + out.status().ToString());
    }
    return out.ValueOrDie();
}

inline std::shared_ptr<arrow::ChunkedArray> widen_to_chunked(const arrow::Datum& d,
                                                             int64_t nrows) {
    switch (d.kind()) {
        case arrow::Datum::CHUNKED_ARRAY:
            return d.chunked_array();
        case arrow::Datum::ARRAY:
            return std::make_shared<arrow::ChunkedArray>(d.make_array());
        case arrow::Datum::SCALAR: {
            auto a = pull(arrow::MakeArrayFromScalar(*d.scalar(), nrows), "MakeArrayFromScalar");
            return std::make_shared<arrow::ChunkedArray>(a);
        }
        default:
            throw std::runtime_error("widen_to_chunked: unsupported Datum kind");
    }
}

inline ColType kind_of(const std::shared_ptr<arrow::ChunkedArray>& c) {
    return arrowTypeToColType(c->type());
}

inline void typecheck_binary(ColType lt, ColType rt, BinaryExpr::Op op) {
    using Op = BinaryExpr::Op;
    switch (op) {
        case Op::ADD:
        case Op::SUB:
        case Op::MUL:
        case Op::DIV:
        case Op::MOD:
            if (!isNumeric(lt) || !isNumeric(rt)) {
                throw std::invalid_argument(
                    "Arithmetic requires numeric operands, got " + colTypeToString(lt) +
                    " and " + colTypeToString(rt));
            }
            (void)promoteTypes(lt, rt);
            break;
        case Op::EQ:
        case Op::NEQ:
            if ((isNumeric(lt) && isNumeric(rt)) || (lt == rt)) return;
            throw std::invalid_argument("Cannot compare " + colTypeToString(lt) + " with " +
                                        colTypeToString(rt));
        case Op::LT:
        case Op::LE:
        case Op::GT:
        case Op::GE:
            if ((isNumeric(lt) && isNumeric(rt)) || (lt == rt && lt == ColType::STRING)) return;
            throw std::invalid_argument("Cannot order " + colTypeToString(lt) + " with " +
                                        colTypeToString(rt));
        case Op::AND:
        case Op::OR:
            if (lt != ColType::BOOLEAN || rt != ColType::BOOLEAN) {
                throw std::invalid_argument(
                    "Boolean op requires boolean operands, got " + colTypeToString(lt) +
                    " and " + colTypeToString(rt));
            }
            break;
    }
}

}  // namespace arx
}  // namespace dfl
