#include "EagerDataFrame.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/compute/api.h>
#include <arrow/compute/initialize.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>

#ifdef DFL_HAVE_PARQUET
#include <parquet/arrow/writer.h>

#endif


namespace dfl
{


namespace {

void ensureComputeInitialized() {
    static const bool init_ok = []() {
        auto s = arrow::compute::Initialize();
        if (!s.ok()) { throw std::runtime_error("arrow::compute::Initialize failed: " + s.ToString());}
        return true;
    }();
    (void)init_ok;
}

void ensureOk(const arrow::Status& s, const std::string& context) {
    if (!s.ok()) {throw std::runtime_error(context + ": " + s.ToString());}
}

template <typename T>
T unwrap(arrow::Result<T>&& res, const std::string& context) {
    if (!res.ok()) {throw std::runtime_error(context + ": " + res.status().ToString());}
    return std::move(res).ValueOrDie();
}

arrow::Datum call(const std::string& fn,const std::vector<arrow::Datum>& args, const arrow::compute::FunctionOptions* opts = nullptr) {
    auto res = arrow::compute::CallFunction(fn, args, opts);
    if (!res.ok()) {throw std::runtime_error("compute::" + fn + ": " + res.status().ToString());}
    return res.ValueOrDie();
}

std::shared_ptr<arrow::ChunkedArray> datumToChunkedArray(
    const arrow::Datum& d, int64_t nrows) {
    switch (d.kind()) {
        case arrow::Datum::CHUNKED_ARRAY:
            return d.chunked_array();
        case arrow::Datum::ARRAY:
            return std::make_shared<arrow::ChunkedArray>(d.make_array());
        case arrow::Datum::SCALAR: {
            auto arr = unwrap(arrow::MakeArrayFromScalar(*d.scalar(), nrows),"MakeArrayFromScalar");
            return std::make_shared<arrow::ChunkedArray>(arr);
        }
        default:
            throw std::runtime_error("datumToChunkedArray: unsupported Datum kind");
    }
}

void validateBinaryOp(ColType lt, ColType rt, BinaryExpr::Op op) {
    using Op = BinaryExpr::Op;
    switch (op) {
        case Op::ADD: case Op::SUB: case Op::MUL: case Op::DIV: case Op::MOD:
            if (!isNumeric(lt) || !isNumeric(rt)) {throw std::invalid_argument( "Arithmetic requires numeric operands, got " +colTypeToString(lt) + " and " + colTypeToString(rt));}
            (void)promoteTypes(lt, rt);
            break;
        case Op::EQ: case Op::NEQ:
            if ((isNumeric(lt) && isNumeric(rt)) ||(lt == rt)) return;
            throw std::invalid_argument("Cannot compare " + colTypeToString(lt) + " with " + colTypeToString(rt));
        case Op::LT: case Op::LE: case Op::GT: case Op::GE:
            if ((isNumeric(lt) && isNumeric(rt)) || (lt == rt && lt == ColType::STRING)) return;
            throw std::invalid_argument("Cannot order " + colTypeToString(lt) + " with " + colTypeToString(rt));
        case Op::AND: case Op::OR:
            if (lt != ColType::BOOLEAN || rt != ColType::BOOLEAN) {
                throw std::invalid_argument("Boolean op requires boolean operands, got " +colTypeToString(lt) + " and " + colTypeToString(rt));
            }
            break;
    }
}

ColType chunkedType(const std::shared_ptr<arrow::ChunkedArray>& c) {
    return arrowTypeToColType(c->type());
}


std::shared_ptr<arrow::ChunkedArray> evalExprOn(
    const std::shared_ptr<arrow::Table>& table,
    const std::shared_ptr<Expr>& expr);

std::shared_ptr<arrow::ChunkedArray> evalColExpr(
    const std::shared_ptr<arrow::Table>& table, const ColExpr& e) {
    
    auto c = table->GetColumnByName(e.name);
    if (!c) throw std::runtime_error("column not found: " + e.name);
    return c;
}

std::shared_ptr<arrow::ChunkedArray> evalLitExpr(
    const std::shared_ptr<arrow::Table>& table, const LitExpr& e) {
    if (!e.value.is_scalar()) {throw std::runtime_error("LitExpr without scalar value");}
    
    auto arr = unwrap(arrow::MakeArrayFromScalar(*e.value.scalar(), table->num_rows()),"LitExpr: MakeArrayFromScalar");
    return std::make_shared<arrow::ChunkedArray>(arr);
}

std::shared_ptr<arrow::ChunkedArray> evalBinaryExpr(
    const std::shared_ptr<arrow::Table>& table, const BinaryExpr& e) {
    auto L = evalExprOn(table, e.left);
    auto R = evalExprOn(table, e.right);
    validateBinaryOp(chunkedType(L), chunkedType(R), e.op);

    if (e.op == BinaryExpr::Op::MOD) {
        std::shared_ptr<arrow::Array> left_arr;
        std::shared_ptr<arrow::Array> right_arr;

        if (L->num_chunks() == 0) {
            left_arr = unwrap(arrow::MakeArrayOfNull(L->type(), table->num_rows()),"mod: empty left fill");
        } else {
            left_arr = unwrap(arrow::Concatenate(L->chunks()), "mod: concat left chunks");
        
        }
        if (R->num_chunks() == 0) {
            right_arr = unwrap(arrow::MakeArrayOfNull(R->type(), table->num_rows()),"mod: empty right fill");
        
        } else {
            right_arr = unwrap(arrow::Concatenate(R->chunks()), "mod: concat right chunks");
        }

        auto out_ty = left_arr->type();
        auto out_builder = unwrap(arrow::MakeBuilder(out_ty), "mod: MakeBuilder");
        ensureOk(out_builder->Reserve(table->num_rows()), "mod: Reserve");

        for (int64_t i = 0; i < table->num_rows(); ++i) {
            auto ls = left_arr->GetScalar(i).ValueOrDie();
            auto rs = right_arr->GetScalar(i).ValueOrDie();

            if (!ls->is_valid || !rs->is_valid) {
                
                ensureOk(out_builder->AppendNull(), "mod: AppendNull");
                continue;
            }

            double lv = std::stod(ls->ToString());
            double rv = std::stod(rs->ToString());
            if (rv == 0.0) {
                ensureOk(out_builder->AppendNull(), "mod: divide-by-zero null");
                continue;
            }

            if (out_ty->id() == arrow::Type::INT32 || out_ty->id() == arrow::Type::INT64) {
                int64_t lvi = static_cast<int64_t>(lv);
                int64_t rvi = static_cast<int64_t>(rv);
                auto out_scalar = std::make_shared<arrow::Int64Scalar>(lvi % rvi);
                ensureOk(out_builder->AppendScalar(*out_scalar), "mod: AppendScalar int");
            } else {
                auto out_scalar = std::make_shared<arrow::DoubleScalar>(std::fmod(lv, rv));
                ensureOk(out_builder->AppendScalar(*out_scalar), "mod: AppendScalar float");
            }
        }

        std::shared_ptr<arrow::Array> out_arr;
        ensureOk(out_builder->Finish(&out_arr), "mod: Finish");
        return std::make_shared<arrow::ChunkedArray>(out_arr);
    }

    const char* fn = nullptr;
    switch (e.op) {
        case BinaryExpr::Op::ADD: fn = "add";  break;
        case BinaryExpr::Op::SUB: fn = "subtract"; break;
        case BinaryExpr::Op::MUL: fn = "multiply";  break;
        case BinaryExpr::Op::DIV: fn = "divide"; break;
        case BinaryExpr::Op::EQ:  fn = "equal"; break;
        case BinaryExpr::Op::NEQ: fn = "not_equal"; break;
        case BinaryExpr::Op::LT:  fn = "less";break;
        case BinaryExpr::Op::LE:  fn = "less_equal";break;
        case BinaryExpr::Op::GT:  fn = "greater"; break;
        case BinaryExpr::Op::GE:  fn = "greater_equal"; break;
        case BinaryExpr::Op::AND: fn = "and_kleene"; break;
        case BinaryExpr::Op::OR:  fn = "or_kleene";  break;
        case BinaryExpr::Op::MOD: fn = nullptr; break;
    }

    auto out = call(fn, { arrow::Datum(L), arrow::Datum(R) });
    return datumToChunkedArray(out, table->num_rows());
}

std::shared_ptr<arrow::ChunkedArray> evalUnaryExpr(
    const std::shared_ptr<arrow::Table>& table, const UnaryExpr& e) {
    auto C = evalExprOn(table, e.child);
    const char* fn = nullptr;
    switch (e.op) {
        case UnaryExpr::Op::NEG:         fn = "negate"; break;
        case UnaryExpr::Op::NOT:
            if (chunkedType(C) != ColType::BOOLEAN) { throw std::invalid_argument("NOT requires a boolean operand");}
            fn = "invert"; break;
        case UnaryExpr::Op::ABS:
            if (!isNumeric(chunkedType(C))) {throw std::invalid_argument("abs() requires a numeric operand");}
            fn = "abs"; break;
        case UnaryExpr::Op::IS_NULL:     fn = "is_null";  break;
        case UnaryExpr::Op::IS_NOT_NULL: fn = "is_valid"; break;
    }
    auto out = call(fn, { arrow::Datum(C) });
    return datumToChunkedArray(out, table->num_rows());
}

std::shared_ptr<arrow::ChunkedArray> evalStringExpr(
    const std::shared_ptr<arrow::Table>& table, const StringExpr& e) {
    auto C = evalExprOn(table, e.child);
    if (chunkedType(C) != ColType::STRING) {throw std::invalid_argument("string function requires a string operand");
    }

    using F = StringExpr::Func;
    switch (e.func) {
        case F::LENGTH: {
            auto out = call("utf8_length", { arrow::Datum(C) });
            return datumToChunkedArray(out, table->num_rows());
        }
        case F::CONTAINS: {
            arrow::compute::MatchSubstringOptions opts(e.arg);
            auto out = call("match_substring", { arrow::Datum(C) }, &opts);
            return datumToChunkedArray(out, table->num_rows());
        }
        case F::STARTS_WITH: {
            arrow::compute::MatchSubstringOptions opts(e.arg);
            auto out = call("starts_with", { arrow::Datum(C) }, &opts);
            return datumToChunkedArray(out, table->num_rows());
        }
        case F::ENDS_WITH: {
            arrow::compute::MatchSubstringOptions opts(e.arg);
            auto out = call("ends_with", { arrow::Datum(C) }, &opts);
            return datumToChunkedArray(out, table->num_rows());
        }
        case F::TO_LOWER: {
            auto out = call("utf8_lower", { arrow::Datum(C) });
            return datumToChunkedArray(out, table->num_rows());
        }
        case F::TO_UPPER: {
            auto out = call("utf8_upper", { arrow::Datum(C) });
            return datumToChunkedArray(out, table->num_rows());
        }
    }
    throw std::runtime_error("unreachable: StringExpr::Func");
}

std::shared_ptr<arrow::ChunkedArray> evalAggExpr(
    const std::shared_ptr<arrow::Table>& table, const AggExpr& e) {
    auto C = evalExprOn(table, e.child);

    const char* fn = nullptr;
    switch (e.func) {
        case AggExpr::Func::SUM:   fn = "sum"; break;
        case AggExpr::Func::MEAN:  fn = "mean"; break;
        case AggExpr::Func::COUNT: fn = "count"; break;
        case AggExpr::Func::MIN:   fn = "min";break;
        case AggExpr::Func::MAX:   fn = "max"; break;
    }

    std::unique_ptr<arrow::compute::FunctionOptions> opts;
    if (e.func == AggExpr::Func::COUNT) {
        opts = std::make_unique<arrow::compute::CountOptions>();
    } else if (e.func == AggExpr::Func::MIN || e.func == AggExpr::Func::MAX || e.func == AggExpr::Func::SUM || e.func == AggExpr::Func::MEAN) {
        opts = std::make_unique<arrow::compute::ScalarAggregateOptions>();
    }

    auto res = arrow::compute::CallFunction(fn, { arrow::Datum(C) }, opts.get());
    if (!res.ok()) {throw std::runtime_error(std::string("aggregate ") + fn + ": " +res.status().ToString());}
    auto scalar = res.ValueOrDie().scalar();
    auto arr = unwrap(arrow::MakeArrayFromScalar(*scalar, 1),"agg: MakeArrayFromScalar");
    return std::make_shared<arrow::ChunkedArray>(arr);
}

std::shared_ptr<arrow::ChunkedArray> evalExprOn(
    const std::shared_ptr<arrow::Table>& table,
    const std::shared_ptr<Expr>& expr) {
    if (!expr)  throw std::runtime_error("evalExpr: null expression");
    if (!table) throw std::runtime_error("evalExpr: null table");

    if (auto c = std::dynamic_pointer_cast<ColExpr>(expr)) return evalColExpr(table, *c);
    if (auto l = std::dynamic_pointer_cast<LitExpr>(expr))  return evalLitExpr(table, *l);
    if (auto a = std::dynamic_pointer_cast<AliasExpr>(expr))  return evalExprOn(table, a->child);
    if (auto b = std::dynamic_pointer_cast<BinaryExpr>(expr)) return evalBinaryExpr(table, *b);
    if (auto u = std::dynamic_pointer_cast<UnaryExpr>(expr)) return evalUnaryExpr(table, *u);
    if (auto s = std::dynamic_pointer_cast<StringExpr>(expr)) return evalStringExpr(table, *s);
    if (auto g = std::dynamic_pointer_cast<AggExpr>(expr)) return evalAggExpr(table, *g);

    throw std::runtime_error("evalExpr: unknown expression node");
}


std::string inferName(const std::shared_ptr<Expr>& e, size_t index) {
    if (auto a = std::dynamic_pointer_cast<AliasExpr>(e)) return a->alias;
    if (auto c = std::dynamic_pointer_cast<ColExpr>(e))   return c->name;
    return "expr_" + std::to_string(index);
}

std::shared_ptr<arrow::Table> makeTable(
    const std::vector<std::string>& names,
    const std::vector<std::shared_ptr<arrow::ChunkedArray>>& cols) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        fields.push_back(arrow::field(names[i], cols[i]->type()));
    }
    return arrow::Table::Make(arrow::schema(fields), cols);
}

} 


EagerDataFrame::EagerDataFrame() : table_(nullptr) {
    ensureComputeInitialized();
}

EagerDataFrame::EagerDataFrame(std::shared_ptr<arrow::Table> table)
    : table_(std::move(table)) {
    ensureComputeInitialized();
}

int64_t EagerDataFrame::num_rows() const { return numRows(); }

int64_t EagerDataFrame::num_columns() const {
    if (!table_) return 0;
    return table_->num_columns();
}

std::vector<std::string> EagerDataFrame::columnNames() const {
    std::vector<std::string> names;
    if (!table_) return names;
    names.reserve(table_->num_columns());
    for (const auto& f : table_->schema()->fields()) names.push_back(f->name());
    return names;
}

std::vector<ColType> EagerDataFrame::columnTypes() const {
    std::vector<ColType> types;
    if (!table_) return types;
    types.reserve(table_->num_columns());
    for (const auto& f : table_->schema()->fields()) {
        types.push_back(arrowTypeToColType(f->type()));
    }
    return types;
}

int64_t EagerDataFrame::numRows() const {
    return table_ ? table_->num_rows() : 0;
}

std::shared_ptr<arrow::Table> EagerDataFrame::table() const { return table_; }

std::shared_ptr<arrow::ChunkedArray> EagerDataFrame::evalExpr(
    const std::shared_ptr<Expr>& expr) const {
    return evalExprOn(table_, expr);
}


void EagerDataFrame::print(int64_t maxRows) const {
    if (!table_) {
        std::cout << "(empty EagerDataFrame)\n";
        return;
    }

    const auto& schema = table_->schema();
    const int   ncols  = schema->num_fields();
    const int64_t nrows = table_->num_rows();
    const int64_t shown = std::min<int64_t>(maxRows, nrows);

    std::vector<std::vector<std::string>> cells(shown, std::vector<std::string>(ncols));
    std::vector<size_t> widths(ncols, 0);
    for (int c = 0; c < ncols; ++c) widths[c] = schema->field(c)->name().size();

    for (int64_t r = 0; r < shown; ++r) {
        for (int c = 0; c < ncols; ++c) {
            auto scalar_res = table_->column(c)->GetScalar(r);
            std::string s;
            if (!scalar_res.ok()) { s = "?"; } 
            else {
                auto sc = scalar_res.ValueOrDie();
                s = (sc && sc->is_valid) ? sc->ToString() : "null";
            }
            cells[r][c] = s;
            widths[c] = std::max(widths[c], s.size());
        }
    }

    auto printRow = [&](const std::vector<std::string>& row) {
        for (int c = 0; c < ncols; ++c) {
            std::cout << std::string(widths[c] - row[c].size(), ' ') << row[c];
            if (c + 1 < ncols) std::cout << " | ";
        }
        std::cout << "\n";
    };

    std::vector<std::string> header(ncols);
    for (int c = 0; c < ncols; ++c) header[c] = schema->field(c)->name();
    printRow(header);

    size_t total = 0;
    for (auto w : widths) total += w;
    total += (ncols > 0 ? (ncols - 1) * 3 : 0);
    std::cout << std::string(total, '-') << "\n";

    for (int64_t r = 0; r < shown; ++r) printRow(cells[r]);

    if (nrows > shown) {std::cout << "... (" << (nrows - shown) << " more rows)\n";}
}


EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& columns) const {
    if (!table_) return EagerDataFrame();

    std::vector<int> indices;
    indices.reserve(columns.size());
    for (const auto& name : columns) {
        int idx = table_->schema()->GetFieldIndex(name);
        if (idx < 0) throw std::runtime_error("select: unknown column '" + name + "'");
        indices.push_back(idx);
    }
    auto out = unwrap(table_->SelectColumns(indices), "select");
    return EagerDataFrame(out);
}

EagerDataFrame EagerDataFrame::select(std::initializer_list<const char*> columns) const {
    return select(std::vector<std::string>(columns.begin(), columns.end()));
}

EagerDataFrame EagerDataFrame::select(const std::vector<ExprBuilder>& exprs) const {
    if (!table_) return EagerDataFrame();

    std::vector<std::string>                          names;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;
    names.reserve(exprs.size());
    cols.reserve(exprs.size());

    for (size_t i = 0; i < exprs.size(); ++i) {
        auto e    = exprs[i].expr();
        auto col  = evalExpr(e);
        names.push_back(inferName(e, i));
        cols.push_back(col);
    }
    return EagerDataFrame(makeTable(names, cols));
}


EagerDataFrame EagerDataFrame::filter(const ExprBuilder& predicate) const {
    if (!table_) return EagerDataFrame();

    auto mask = evalExpr(predicate.expr());
    if (chunkedType(mask) != ColType::BOOLEAN) {
        throw std::invalid_argument("filter: predicate must be a boolean expression");
    }
    auto out = call("filter", { arrow::Datum(table_), arrow::Datum(mask) });
    return EagerDataFrame(out.table());
}


EagerDataFrame EagerDataFrame::with_column(const std::string& name, const ExprBuilder& expr) const {
    if (!table_) throw std::runtime_error("with_column: no table");

    auto col   = evalExpr(expr.expr());
    auto field = arrow::field(name, col->type());

    const int idx = table_->schema()->GetFieldIndex(name);
    if (idx < 0) {
        auto out = unwrap(table_->AddColumn(table_->num_columns(), field, col),"with_column: add");
        return EagerDataFrame(out);
    } else {
        auto out = unwrap(table_->SetColumn(idx, field, col), "with_column: set");
        return EagerDataFrame(out);
    }
}


EagerDataFrame EagerDataFrame::group_by(const std::vector<std::string>& keys) const {
    EagerDataFrame out(table_);
    out.group_keys_ = keys;
    return out;
}


namespace {

bool keysDiffer(const std::vector<std::shared_ptr<arrow::Array>>& keys, int64_t i, int64_t j) {
    for (const auto& k : keys) {
        auto si = k->GetScalar(i).ValueOrDie();
        auto sj = k->GetScalar(j).ValueOrDie();
        if (!si->Equals(*sj)) return true;
    }
    return false;
}

std::vector<std::shared_ptr<arrow::Array>> combinedKeyArrays(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& keys) {
    std::vector<std::shared_ptr<arrow::Array>> out;
    out.reserve(keys.size());
    for (const auto& k : keys) {
        auto c = table->GetColumnByName(k);
        if (!c) throw std::runtime_error("group_by: unknown column '" + k + "'");
        auto combined = unwrap(arrow::Concatenate(c->chunks()),"group_by: concatenate key '" + k + "'");
        out.push_back(combined);
    }
    return out;
}

} 

EagerDataFrame EagerDataFrame::aggregate(
    const std::map<std::string, ExprBuilder>& aggMap) const {
    if (!table_) return EagerDataFrame();

    if (group_keys_.empty()) {
        std::vector<std::string>                          names;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;
        names.reserve(aggMap.size());
        cols.reserve(aggMap.size());
        for (const auto& [name, eb] : aggMap) {
            names.push_back(name);
            cols.push_back(evalExpr(eb.expr()));
        }
        return EagerDataFrame(makeTable(names, cols));
    }

    auto sorted_df = this->sort(group_keys_, true);
    sorted_df.group_keys_.clear();
    auto sorted_table = sorted_df.table();

    auto key_arrays = combinedKeyArrays(sorted_table, group_keys_);
    const int64_t n = sorted_table->num_rows();

    std::vector<std::pair<int64_t, int64_t>> groups;
    if (n > 0) {
        int64_t start = 0;
        for (int64_t i = 1; i < n; ++i) {
            if (keysDiffer(key_arrays, i - 1, i)) {
                groups.emplace_back(start, i);
                start = i;
            }
        }
        groups.emplace_back(start, n);
    }

    const size_t k = group_keys_.size();
    const size_t a = aggMap.size();

    std::vector<std::vector<std::shared_ptr<arrow::Scalar>>> out_cols(k + a);
    for (auto& col : out_cols) col.reserve(groups.size());

    std::vector<std::pair<std::string, ExprBuilder>> aggList(
        aggMap.begin(), aggMap.end());

    for (const auto& [gs, ge] : groups) {
        for (size_t ki = 0; ki < k; ++ki) {
            out_cols[ki].push_back(key_arrays[ki]->GetScalar(gs).ValueOrDie());
        }
        auto slice = sorted_table->Slice(gs, ge - gs);
        for (size_t ai = 0; ai < a; ++ai) {
            auto chunked = evalExprOn(slice, aggList[ai].second.expr());
            out_cols[k + ai].push_back(chunked->GetScalar(0).ValueOrDie());
        }
    }

    std::vector<std::string>                          names;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunked_cols;
    names.reserve(k + a);
    chunked_cols.reserve(k + a);

    auto scalarsToChunked = [](const std::vector<std::shared_ptr<arrow::Scalar>>& v)
        -> std::shared_ptr<arrow::ChunkedArray> {
        if (v.empty()) {
            return std::make_shared<arrow::ChunkedArray>(std::vector<std::shared_ptr<arrow::Array>>{}, arrow::null());
        }
        auto ty      = v.front()->type;
        auto builder = unwrap(arrow::MakeBuilder(ty), "MakeBuilder");
        ensureOk(builder->Reserve(static_cast<int64_t>(v.size())), "Reserve agg builder");
        for (const auto& s : v) ensureOk(builder->AppendScalar(*s), "AppendScalar");
        std::shared_ptr<arrow::Array> arr;
        ensureOk(builder->Finish(&arr), "Finish agg builder");
        return std::make_shared<arrow::ChunkedArray>(arr);
    };

    for (size_t ki = 0; ki < k; ++ki) {
        names.push_back(group_keys_[ki]);
        chunked_cols.push_back(scalarsToChunked(out_cols[ki]));
    }
    for (size_t ai = 0; ai < a; ++ai) {
        names.push_back(aggList[ai].first);
        chunked_cols.push_back(scalarsToChunked(out_cols[k + ai]));
    }

    return EagerDataFrame(makeTable(names, chunked_cols));
}

EagerDataFrame EagerDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& aggs) const {
    std::map<std::string, ExprBuilder> agg_map;

    for (const auto& [col_name, fn_name_raw] : aggs) {
        std::string fn_name = fn_name_raw;
        std::transform(fn_name.begin(), fn_name.end(), fn_name.begin(),[](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        ExprBuilder expr;
        if (fn_name == "sum") expr = col(col_name).sum();
        else if (fn_name == "mean") expr = col(col_name).mean();
        else if (fn_name == "count") expr = col(col_name).count();
        else if (fn_name == "min") expr = col(col_name).min();
        else if (fn_name == "max") expr = col(col_name).max();
        else {throw std::invalid_argument("aggregate: unsupported function '" + fn_name_raw + "'");}

        std::string out_name = col_name + "_" + fn_name;
        if (agg_map.find(out_name) != agg_map.end()) {
            int suffix = 2;
            std::string candidate;
            do {candidate = out_name + "_" + std::to_string(suffix++);} 
            while (agg_map.find(candidate) != agg_map.end());
            out_name = candidate;
        }

        agg_map.emplace(std::move(out_name), std::move(expr));
    }

    return aggregate(agg_map);
}


EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& columns,bool ascending) const {
    if (!table_) return EagerDataFrame();

    std::vector<arrow::compute::SortKey> keys;
    keys.reserve(columns.size());
    for (const auto& c : columns) {
        keys.emplace_back(c, ascending ? arrow::compute::SortOrder::Ascending: arrow::compute::SortOrder::Descending);
    }
    arrow::compute::SortOptions opts(keys);

    auto indices = call("sort_indices", { arrow::Datum(table_) }, &opts);
    auto taken   = call("take", { arrow::Datum(table_), indices });
    return EagerDataFrame(taken.table());
}


EagerDataFrame EagerDataFrame::head(int64_t n) const {
    if (!table_) return EagerDataFrame();
    const int64_t take = std::min<int64_t>(n, table_->num_rows());
    return EagerDataFrame(table_->Slice(0, take));
}


namespace {

struct HashRow {
    std::string key;
    bool        has_null = false;
};

HashRow rowKey(const std::vector<std::shared_ptr<arrow::Array>>& keys,int64_t row) {
    HashRow r;
    std::ostringstream oss;
    for (const auto& k : keys) {
        if (k->IsNull(row)) {
            r.has_null = true;
            return r;
        }
        auto scalar = k->GetScalar(row).ValueOrDie();
        oss << scalar->ToString() << '\x1f'; 
    }
    r.key = oss.str();
    return r;
}

void appendRow(const std::shared_ptr<arrow::Table>& src, int64_t row,std::vector<std::unique_ptr<arrow::ArrayBuilder>>& builders) {
    for (int c = 0; c < src->num_columns(); ++c) {
        auto col = src->column(c);
        auto sc  = col->GetScalar(row).ValueOrDie();
        ensureOk(builders[c]->AppendScalar(*sc), "join: AppendScalar");
    }
}

void appendNulls(std::vector<std::unique_ptr<arrow::ArrayBuilder>>& builders) {
    for (auto& b : builders) ensureOk(b->AppendNull(), "join: AppendNull");
}

std::vector<std::unique_ptr<arrow::ArrayBuilder>> buildersForSchema(
    const std::shared_ptr<arrow::Schema>& schema) {
    std::vector<std::unique_ptr<arrow::ArrayBuilder>> out;
    out.reserve(schema->num_fields());
    for (const auto& f : schema->fields()) {
        out.push_back(unwrap(arrow::MakeBuilder(f->type()), "MakeBuilder"));
    }
    return out;
}

std::vector<std::shared_ptr<arrow::Field>> excludeKeys(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::unordered_set<std::string>& drop) {
    std::vector<std::shared_ptr<arrow::Field>> fs;
    for (const auto& f : schema->fields()) {
        if (drop.count(f->name())) continue;
        fs.push_back(f);
    }
    return fs;
}

} 

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other,const std::vector<std::string>& on,const std::string& how) const {
    if (!table_ || !other.table_) {
        throw std::runtime_error("join: null table");
    }
    if (on.empty()) throw std::invalid_argument("join: empty key list");
    if (how != "inner" && how != "left" && how != "right" && how != "outer") {
        throw std::invalid_argument("join: unsupported join type '" + how + "'");
    }

    auto concatChunks = [](const std::shared_ptr<arrow::Table>& t,const std::vector<std::string>& names) {
        std::vector<std::shared_ptr<arrow::Array>> out;
        out.reserve(names.size());
        for (const auto& n : names) {
            auto c = t->GetColumnByName(n);
            if (!c) throw std::runtime_error("join: missing key '" + n + "'");
            if (c->num_chunks() == 0) {out.push_back(unwrap(arrow::MakeArrayOfNull(c->type(), t->num_rows()),"join: empty key fill '" + n + "'"));} 
            else { out.push_back(unwrap(arrow::Concatenate(c->chunks()),"join: concat key '" + n + "'"));}
        }
        return out;
    };

    auto left_keys = concatChunks(table_,        on);
    auto right_keys = concatChunks(other.table_,  on);

    std::unordered_map<std::string, std::vector<int64_t>> rhs_index;
    rhs_index.reserve(static_cast<size_t>(other.table_->num_rows()));
    for (int64_t r = 0; r < other.table_->num_rows(); ++r) {
        auto k = rowKey(right_keys, r);
        if (k.has_null) continue;
        rhs_index[k.key].push_back(r);
    }

    std::unordered_set<std::string> keySet(on.begin(), on.end());
    auto right_extra_fields = excludeKeys(other.table_->schema(), keySet);

    std::vector<std::shared_ptr<arrow::Field>> out_fields = table_->schema()->fields();
    for (const auto& f : right_extra_fields) out_fields.push_back(f);
    auto out_schema = arrow::schema(out_fields);

    auto left_builders  = buildersForSchema(table_->schema());
    std::vector<std::unique_ptr<arrow::ArrayBuilder>> right_extra_builders;
    right_extra_builders.reserve(right_extra_fields.size());
    for (const auto& f : right_extra_fields) {
        right_extra_builders.push_back(unwrap(arrow::MakeBuilder(f->type()), "MakeBuilder"));
    }

    std::vector<uint8_t> right_matched(other.table_->num_rows(), 0);

    auto appendRightExtras = [&](int64_t r) {
        int bi = 0;
        for (int c = 0; c < other.table_->num_columns(); ++c) {
            const auto& name = other.table_->schema()->field(c)->name();
            if (keySet.count(name)) continue;
            auto sc = other.table_->column(c)->GetScalar(r).ValueOrDie();
            ensureOk(right_extra_builders[bi++]->AppendScalar(*sc),"join: right AppendScalar");
        }
    };
    auto appendRightExtraNulls = [&]() {
        for (auto& b : right_extra_builders) {
            ensureOk(b->AppendNull(), "join: right AppendNull");
        }
    };

    for (int64_t l = 0; l < table_->num_rows(); ++l) {
        auto k = rowKey(left_keys, l);
        if (k.has_null) {
            if (how == "left" || how == "outer") {
                appendRow(table_, l, left_builders);
                appendRightExtraNulls();
            }
            continue;
        }
        auto it = rhs_index.find(k.key);
        if (it == rhs_index.end()) {
            if (how == "left" || how == "outer") {
                appendRow(table_, l, left_builders);
                appendRightExtraNulls();
            }
        } else {
            for (int64_t r : it->second) {
                appendRow(table_, l, left_builders);
                appendRightExtras(r);
                right_matched[r] = 1;
            }
        }
    }

    if (how == "right" || how == "outer") {
        for (int64_t r = 0; r < other.table_->num_rows(); ++r) {
            if (right_matched[r]) continue;
            for (int c = 0; c < table_->num_columns(); ++c) {
                const auto& name = table_->schema()->field(c)->name();
                if (keySet.count(name)) {
                    auto sc = other.table_->GetColumnByName(name)->GetScalar(r).ValueOrDie();
                    ensureOk(left_builders[c]->AppendScalar(*sc),"join: right-only AppendScalar");
                } else {
                    ensureOk(left_builders[c]->AppendNull(),"join: right-only AppendNull");
                }
            }
            appendRightExtras(r);
        }
    }

    std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;
    cols.reserve(out_fields.size());
    for (auto& b : left_builders) {
        std::shared_ptr<arrow::Array> a;
        ensureOk(b->Finish(&a), "join: left Finish");
        cols.push_back(std::make_shared<arrow::ChunkedArray>(a));
    }
    for (auto& b : right_extra_builders) {
        std::shared_ptr<arrow::Array> a;
        ensureOk(b->Finish(&a), "join: right Finish");
        cols.push_back(std::make_shared<arrow::ChunkedArray>(a));
    }

    return EagerDataFrame(arrow::Table::Make(out_schema, cols));
}


void EagerDataFrame::write_csv(const std::string& path) const {
    if (!table_) throw std::runtime_error("write_csv: no table");

    auto outfile = unwrap(arrow::io::FileOutputStream::Open(path),"write_csv: open '" + path + "'");
    auto opts = arrow::csv::WriteOptions::Defaults();
    ensureOk(arrow::csv::WriteCSV(*table_, opts, outfile.get()),"write_csv: WriteCSV");
    ensureOk(outfile->Close(), "write_csv: close");
}

void EagerDataFrame::write_parquet(const std::string& path) const {
#ifdef DFL_HAVE_PARQUET
    if (!table_) throw std::runtime_error("write_parquet: no table");

    auto outfile = unwrap(arrow::io::FileOutputStream::Open(path),"write_parquet: open '" + path + "'");
    ensureOk(parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(),outfile, 64 * 1024),"write_parquet: WriteTable");
    ensureOk(outfile->Close(), "write_parquet: close");
#else
    (void)path;
    throw std::runtime_error("write_parquet: DataFrameLib was built without Parquet support");
#endif
}

} 
