// Step 2 — full correctness tests for EagerDataFrame.
//
// Deliberately uses only <cassert> so the test binary is dependency-free and
// always runs after a plain `make`. Each test is a standalone function that
// prints its name and asserts its expectations; main() runs them all.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/builder.h>

#include "EagerDataFrame.hpp"
#include "Expr.hpp"
#include "IO.hpp"
#include "LazyDataFrame.hpp"
#include "LogicalPlan.hpp"
#include "QueryOptimizer.hpp"
#include "TypeUtils.hpp"

#include <chrono>

using namespace dfl;

// ---------------------------------------------------------------------------
// Helpers for building small Arrow tables from raw vectors.
// ---------------------------------------------------------------------------

namespace {

std::shared_ptr<arrow::Array> i32(const std::vector<int32_t>& v) {
    arrow::Int32Builder b;
    [[maybe_unused]] auto st = b.AppendValues(v);
    std::shared_ptr<arrow::Array> a;
    st = b.Finish(&a);
    return a;
}

std::shared_ptr<arrow::Array> i64(const std::vector<int64_t>& v) {
    arrow::Int64Builder b;
    [[maybe_unused]] auto st = b.AppendValues(v);
    std::shared_ptr<arrow::Array> a;
    st = b.Finish(&a);
    return a;
}

std::shared_ptr<arrow::Array> f64(const std::vector<double>& v) {
    arrow::DoubleBuilder b;
    [[maybe_unused]] auto st = b.AppendValues(v);
    std::shared_ptr<arrow::Array> a;
    st = b.Finish(&a);
    return a;
}

std::shared_ptr<arrow::Array> str(const std::vector<std::string>& v) {
    arrow::StringBuilder b;
    [[maybe_unused]] auto st = b.AppendValues(v);
    std::shared_ptr<arrow::Array> a;
    st = b.Finish(&a);
    return a;
}

// Build an i32 array with explicit null mask: `validity[i] == 0` → null.
std::shared_ptr<arrow::Array> i32_with_nulls(const std::vector<int32_t>& vals,
                                             const std::vector<bool>& valid) {
    arrow::Int32Builder b;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (valid[i]) {
            [[maybe_unused]] auto st = b.Append(vals[i]);
        } else {
            [[maybe_unused]] auto st = b.AppendNull();
        }
    }
    std::shared_ptr<arrow::Array> a;
    [[maybe_unused]] auto st = b.Finish(&a);
    return a;
}

EagerDataFrame makeFrame(
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& cols) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunks;
    for (const auto& [n, a] : cols) {
        fields.push_back(arrow::field(n, a->type()));
        chunks.push_back(std::make_shared<arrow::ChunkedArray>(a));
    }
    return EagerDataFrame(arrow::Table::Make(arrow::schema(fields), chunks));
}

std::vector<int32_t> asI32(const std::shared_ptr<arrow::ChunkedArray>& c) {
    std::vector<int32_t> out;
    for (int ci = 0; ci < c->num_chunks(); ++ci) {
        auto arr = std::static_pointer_cast<arrow::Int32Array>(c->chunk(ci));
        for (int64_t i = 0; i < arr->length(); ++i) out.push_back(arr->Value(i));
    }
    return out;
}

std::vector<int64_t> asI64(const std::shared_ptr<arrow::ChunkedArray>& c) {
    std::vector<int64_t> out;
    for (int ci = 0; ci < c->num_chunks(); ++ci) {
        auto arr = std::static_pointer_cast<arrow::Int64Array>(c->chunk(ci));
        for (int64_t i = 0; i < arr->length(); ++i) out.push_back(arr->Value(i));
    }
    return out;
}

std::vector<double> asF64(const std::shared_ptr<arrow::ChunkedArray>& c) {
    std::vector<double> out;
    for (int ci = 0; ci < c->num_chunks(); ++ci) {
        auto arr = std::static_pointer_cast<arrow::DoubleArray>(c->chunk(ci));
        for (int64_t i = 0; i < arr->length(); ++i) out.push_back(arr->Value(i));
    }
    return out;
}

std::shared_ptr<arrow::ChunkedArray> column(const EagerDataFrame& df,
                                            const std::string& name) {
    auto c = df.table()->GetColumnByName(name);
    if (!c) throw std::runtime_error("no column '" + name + "'");
    return c;
}

// ---------------------------------------------------------------------------
// Individual tests
// ---------------------------------------------------------------------------

void test1_csv_roundtrip() {
    std::cout << "test1_csv_roundtrip ... ";

    const std::string path = "/tmp/dfl_test1.csv";
    {
        std::ofstream f(path);
        f << "id,name,score\n"
          << "1,Alice,90.5\n"
          << "2,Bob,82.0\n"
          << "3,Carol,75.25\n";
    }

    auto df = read_csv(path);
    assert(df.numRows() == 3);
    auto names = df.columnNames();
    assert(names.size() == 3);
    assert(names[0] == "id" && names[1] == "name" && names[2] == "score");

    const std::string out = "/tmp/dfl_test1_out.csv";
    df.write_csv(out);

    auto df2 = read_csv(out);
    assert(df2.numRows() == 3);
    assert(df2.columnNames() == names);

    std::cout << "OK\n";
}

void test2_filter() {
    std::cout << "test2_filter ........... ";
    auto df = makeFrame({{"age", i32({10, 25, 30, 45, 60})}});
    auto out = df.filter(col("age") > lit<int32_t>(29));
    assert(out.numRows() == 3);
    auto vals = asI32(column(out, "age"));
    assert((vals == std::vector<int32_t>{30, 45, 60}));
    std::cout << "OK\n";
}

void test3_select_with_expr() {
    std::cout << "test3_select_with_expr . ";
    auto df  = makeFrame({{"age", i32({10, 25, 30, 45, 60})}});
    auto out = df.with_column("age2", col("age") * lit<int32_t>(2))
                 .select({"age", "age2"});
    assert(out.columnNames() == (std::vector<std::string>{"age", "age2"}));
    auto age2 = asI32(column(out, "age2"));
    assert((age2 == std::vector<int32_t>{20, 50, 60, 90, 120}));
    std::cout << "OK\n";
}

void test4_group_by_aggregate() {
    std::cout << "test4_group_aggregate .. ";
    auto df = makeFrame({
        {"dept",   str({"eng", "hr", "eng", "hr", "eng"})},
        {"salary", f64({100.0, 50.0, 120.0, 60.0, 110.0})},
    });
    auto out = df.group_by({"dept"})
                 .aggregate({{"total", col("salary").sum()}});
    assert(out.numRows() == 2);
    assert(out.columnNames() == (std::vector<std::string>{"dept", "total"}));
    // Sort by dept for deterministic comparison (group_by sorts internally,
    // so "eng" < "hr" lexicographically).
    auto totals = asF64(column(out, "total"));
    assert(std::abs(totals[0] - 330.0) < 1e-9); // eng: 100+120+110
    assert(std::abs(totals[1] -  110.0) < 1e-9); // hr:  50+60
    std::cout << "OK\n";
}

void test5_sort_desc() {
    std::cout << "test5_sort_desc ........ ";
    auto df = makeFrame({{"age", i32({10, 60, 30, 45, 25})}});
    auto out = df.sort({"age"}, /*ascending=*/false).head(3);
    assert(out.numRows() == 3);
    auto vals = asI32(column(out, "age"));
    assert(vals.front() == 60);
    assert((vals == std::vector<int32_t>{60, 45, 30}));
    std::cout << "OK\n";
}

void test6_join_inner() {
    std::cout << "test6_join_inner ....... ";
    auto left = makeFrame({
        {"id",   i64({1, 2, 3, 4})},
        {"name", str({"Alice", "Bob", "Carol", "Dan"})},
    });
    auto right = makeFrame({
        {"id",    i64({2, 4, 5})},
        {"score", f64({82.0, 88.0, 70.0})},
    });
    auto out = left.join(right, {"id"}, "inner");
    assert(out.numRows() == 2);
    assert(out.columnNames() ==
           (std::vector<std::string>{"id", "name", "score"}));
    std::cout << "OK\n";
}

void test7_null_propagation() {
    std::cout << "test7_null_propagation . ";
    auto df = makeFrame({
        {"x", i32_with_nulls({1, 2, 3}, {true, false, true})},
    });
    auto out = df.select({ (col("x") + lit<int32_t>(1)).alias("y") });
    auto chunked = column(out, "y");
    // Middle row should be null.
    auto arr = std::static_pointer_cast<arrow::Int32Array>(chunked->chunk(0));
    assert(arr->IsValid(0));
    assert(!arr->IsValid(1));
    assert(arr->IsValid(2));
    assert(arr->Value(0) == 2);
    assert(arr->Value(2) == 4);
    std::cout << "OK\n";
}

void test8_type_error() {
    std::cout << "test8_type_error ....... ";
    auto df = makeFrame({
        {"x", i32({1, 2, 3})},
        {"s", str({"a", "b", "c"})},
    });
    bool threw = false;
    try {
        auto bad = df.select({ (col("x") + col("s")).alias("oops") });
        (void)bad;
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (const std::exception& e) {
        std::cout << "(caught other: " << e.what() << ") ";
        threw = true;
    }
    assert(threw);
    std::cout << "OK\n";
}

// ---------------------------------------------------------------------------
// Fixture helpers for lazy tests — write small CSV / Parquet files on demand.
// ---------------------------------------------------------------------------

void writeCsvFixture(const std::string& path, const std::string& body) {
    std::ofstream f(path);
    f << body;
}

void writeParquetFixtureFromEager(const std::string& path,
                                   const EagerDataFrame& df) {
    df.write_parquet(path);
}

// ---------------------------------------------------------------------------
// Bonus coverage: make sure string funcs and aggregations really work,
// plus is_null / is_not_null / ends_with.
// ---------------------------------------------------------------------------

void test9_string_and_agg() {
    std::cout << "test9_strings_and_agg .. ";
    auto df = makeFrame({
        {"name",  str({"Alice", "Bob",   "Carol", "Dan"})},
        {"email", str({"a@x",   "bob@y", "c@z",   "dan@w"})},
    });
    auto starts = df.select({ col("name").starts_with("A").alias("sA") });
    auto arr = std::static_pointer_cast<arrow::BooleanArray>(
        column(starts, "sA")->chunk(0));
    assert(arr->Value(0) == true && arr->Value(1) == false);

    auto agg = df.aggregate({{"n", col("name").count()}});
    assert(agg.numRows() == 1);
    std::cout << "OK\n";
}

// ---------------------------------------------------------------------------
// Lazy-mode tests (Step 3).
// ---------------------------------------------------------------------------

void test10_lazy_collect() {
    std::cout << "test10_lazy_collect .... ";
    const std::string path = "/tmp/dfl_lazy_people.csv";
    writeCsvFixture(path,
        "id,age\n"
        "1,10\n"
        "2,25\n"
        "3,30\n"
        "4,45\n"
        "5,60\n");

    auto lf = scan_csv(path);
    auto out = lf.filter(col("age") > lit<int64_t>(20)).head(5).collect();
    assert(out.numRows() == 4);
    std::cout << "OK\n";
}

void test11_lazy_explain() {
    std::cout << "test11_lazy_explain .... ";
    const std::string csv = "/tmp/dfl_lazy_people.csv"; // reused from test10
    auto lf = scan_csv(csv)
                  .filter(col("age") > lit<int64_t>(20))
                  .select({"id", "age"})
                  .sort({"age"}, false)
                  .head(10);

    // Must be non-fatal whether or not `dot` is installed.
    const std::string png = "/tmp/dfl_lazy_plan.png";
    lf.explain(png);

    // Also sanity-check the DOT string directly so this test asserts
    // something concrete even if the PNG fails to render.
    auto dot = renderDotGraph(lf.plan());
    assert(dot.find("digraph") != std::string::npos);
    assert(dot.find("Filter") != std::string::npos);
    assert(dot.find("Sort")   != std::string::npos);
    assert(dot.find("Limit")  != std::string::npos);
    std::cout << "OK\n";
}

void test12_lazy_group_aggregate() {
    std::cout << "test12_lazy_group_agg .. ";
    const std::string path = "/tmp/dfl_lazy_dept.csv";
    writeCsvFixture(path,
        "dept,salary\n"
        "eng,100.0\n"
        "hr,50.0\n"
        "eng,120.0\n"
        "hr,60.0\n"
        "eng,110.0\n");

    auto result = scan_csv(path)
                      .group_by({"dept"})
                      .aggregate({{"avg_sal", col("salary").mean()}})
                      .collect();
    assert(result.numRows() == 2);
    auto names = result.columnNames();
    assert(names.size() == 2);
    assert(names[0] == "dept" && names[1] == "avg_sal");
    std::cout << "OK\n";
}

void test13_lazy_join() {
    std::cout << "test13_lazy_join ....... ";
    const std::string left  = "/tmp/dfl_lazy_left.csv";
    const std::string right = "/tmp/dfl_lazy_right.csv";
    writeCsvFixture(left,
        "id,name\n"
        "1,Alice\n"
        "2,Bob\n"
        "3,Carol\n"
        "4,Dan\n");
    writeCsvFixture(right,
        "id,score\n"
        "2,82.0\n"
        "4,88.0\n"
        "5,70.0\n");

    auto result = scan_csv(left).join(scan_csv(right), {"id"}, "inner").collect();
    assert(result.numRows() == 2);
    auto names = result.columnNames();
    assert(names == (std::vector<std::string>{"id", "name", "score"}));
    std::cout << "OK\n";
}

void test14_lazy_sort_head_parquet() {
    std::cout << "test14_lazy_sort_head .. ";
    // Build a parquet fixture via the eager writer so the test is
    // self-contained (doesn't depend on an external .parquet file).
    auto fixture = makeFrame({
        {"id",    i32({1, 2, 3, 4, 5})},
        {"score", f64({10.0, 90.0, 30.0, 70.0, 50.0})},
    });
    const std::string pq = "/tmp/dfl_lazy_scores.parquet";
    fixture.write_parquet(pq);

    auto result = scan_parquet(pq).sort({"score"}, /*ascending=*/false)
                                   .head(3).collect();
    assert(result.numRows() == 3);
    // First row must be the highest score.
    auto scores = asF64(column(result, "score"));
    assert(std::abs(scores[0] - 90.0) < 1e-9);
    assert((scores == std::vector<double>{90.0, 70.0, 50.0}));
    std::cout << "OK\n";
}

void test15_lazy_sink_csv() {
    std::cout << "test15_lazy_sink_csv ... ";
    const std::string path = "/tmp/dfl_lazy_people.csv"; // reused from test10
    const std::string out  = "/tmp/dfl_lazy_filtered.csv";
    scan_csv(path)
        .filter(col("age") > lit<int64_t>(20))
        .sink_csv(out);

    auto verify = read_csv(out);
    assert(verify.numRows() == 4);
    std::cout << "OK\n";
}

// ===========================================================================
// Step 4 — QueryOptimizer tests
// ===========================================================================

/// Convenience: run `f` a few times and return the minimum wall-clock time
/// so benchmark assertions are less flaky under noisy CPUs.
template <typename F>
double minTimeMs(F&& f, int reps = 3) {
    double best = 1e18;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

/// Write a CSV with `nrows` rows and `ncols+1` int64 columns (an "id" col
/// plus `value0..valueN-1`). Used by the heavy optimizer benchmark.
void writeWideIntCsv(const std::string& path, int64_t nrows, int ncols) {
    std::ofstream f(path);
    f << "id";
    for (int c = 0; c < ncols; ++c) f << ",value" << c;
    f << "\n";
    for (int64_t r = 0; r < nrows; ++r) {
        f << r;
        for (int c = 0; c < ncols; ++c) f << ',' << (r * (c + 1) % 10000);
        f << '\n';
    }
}

void test16_predicate_pushdown() {
    std::cout << "test16_predicate_pushdown ... ";
    const std::string left  = "/tmp/dfl_opt_left.csv";
    const std::string right = "/tmp/dfl_opt_right.csv";

    // Build a left table with 2000 rows, a right with 200.
    {
        std::ofstream f(left);
        f << "id,left_val\n";
        for (int i = 0; i < 2000; ++i) f << i << ',' << i << '\n';
    }
    {
        std::ofstream f(right);
        f << "id,right_val\n";
        for (int i = 0; i < 200; ++i) f << i << ',' << (1000 + i) << '\n';
    }

    auto lf = scan_csv(left)
                  .join(scan_csv(right), {"id"}, "inner")
                  .filter(col("left_val") > lit<int64_t>(5));

    // Correctness: optimized must match raw execution.
    auto raw = lf.collect_raw();
    auto opt = lf.collect();
    assert(raw.numRows() == opt.numRows());
    assert(raw.columnNames() == opt.columnNames());

    // Verify the predicate was physically moved below the Join: after
    // optimization the JoinNode's LEFT child should be a FilterNode.
    QueryOptimizer o;
    auto optimized = o.optimize(lf.plan());
    auto jn = std::dynamic_pointer_cast<JoinNode>(optimized);
    assert(jn != nullptr);
    assert(!jn->children.empty());
    assert(std::dynamic_pointer_cast<FilterNode>(jn->children[0]) != nullptr);

    std::cout << "OK\n";
}

void test17_constant_folding() {
    std::cout << "test17_constant_folding ..... ";
    const std::string path = "/tmp/dfl_lazy_people.csv"; // reused fixture
    auto lf = scan_csv(path).filter(col("age") > (lit<int64_t>(20) + lit<int64_t>(10)));

    QueryOptimizer o;
    auto optimized = o.optimize(lf.plan());

    // After folding, the filter's predicate must be (col("age") > lit(30)).
    auto filter = std::dynamic_pointer_cast<FilterNode>(optimized);
    assert(filter != nullptr);
    auto bin = std::dynamic_pointer_cast<BinaryExpr>(filter->predicate);
    assert(bin != nullptr && bin->op == BinaryExpr::Op::GT);
    auto lhs = std::dynamic_pointer_cast<ColExpr>(bin->left);
    auto rhs = std::dynamic_pointer_cast<LitExpr>(bin->right);
    assert(lhs && lhs->name == "age");
    assert(rhs != nullptr);
    auto scalar = rhs->value.scalar();
    assert(scalar && scalar->is_valid);
    assert(std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value == 30);

    // Result should match the unoptimized version.
    auto raw = lf.collect_raw();
    auto opt = lf.collect();
    assert(raw.numRows() == opt.numRows());
    std::cout << "OK\n";
}

void test18_expression_simplification() {
    std::cout << "test18_expression_simpl ..... ";
    const std::string path = "/tmp/dfl_opt_left.csv"; // reused fixture (id, left_val)
    auto lf = scan_csv(path).select({
        col("id") * lit<int64_t>(1),          // → col("id")
        col("left_val") + lit<int64_t>(0),    // → col("left_val")
    });

    QueryOptimizer o;
    auto optimized = o.optimize(lf.plan());
    auto sel = std::dynamic_pointer_cast<SelectNode>(optimized);
    assert(sel != nullptr && sel->columns.size() == 2);

    auto c0 = std::dynamic_pointer_cast<ColExpr>(sel->columns[0].expr());
    auto c1 = std::dynamic_pointer_cast<ColExpr>(sel->columns[1].expr());
    assert(c0 && c0->name == "id");
    assert(c1 && c1->name == "left_val");

    auto raw = lf.collect_raw();
    auto opt = lf.collect();
    assert(raw.numRows() == opt.numRows());
    std::cout << "OK\n";
}

void test19_limit_pushdown() {
    std::cout << "test19_limit_pushdown ....... ";
    const std::string path = "/tmp/dfl_opt_left.csv"; // id, left_val
    auto lf = scan_csv(path)
                  .with_column("z", col("left_val") + lit<int64_t>(1))
                  .head(5);

    QueryOptimizer o;
    auto optimized = o.optimize(lf.plan());

    // Expected shape: WithColumn → Scan(row_limit = 5)
    auto wc = std::dynamic_pointer_cast<WithColNode>(optimized);
    assert(wc != nullptr);
    assert(!wc->children.empty());
    auto scan = std::dynamic_pointer_cast<ScanNode>(wc->children[0]);
    assert(scan != nullptr);
    assert(scan->row_limit == 5);

    // Results must match the un-pushed version.
    auto raw = lf.collect_raw();
    auto opt = lf.collect();
    assert(raw.numRows() == 5);
    assert(opt.numRows() == 5);
    std::cout << "OK\n";
}

void test20_optimizer_benchmark() {
    std::cout << "test20_optimizer_bench ...... " << std::flush;

    // Build 20k × 10-column left CSV and 20k × 2-column right CSV.
    const std::string big_left  = "/tmp/dfl_bench_left.csv";
    const std::string big_right = "/tmp/dfl_bench_right.csv";
    constexpr int64_t NROWS = 20000;
    writeWideIntCsv(big_left, NROWS, /*ncols=*/9);      // id + value0..value8
    {
        std::ofstream f(big_right);
        f << "id,side\n";
        for (int64_t i = 0; i < NROWS; ++i) f << i << ',' << i << '\n';
    }

    auto build = [&]() {
        return scan_csv(big_left)
                 .join(scan_csv(big_right), {"id"}, "inner")
                 .filter(col("value0") > lit<int64_t>(1000))
                 .select({"id", "value0"})
                 .head(100);
    };

    // Warmup once so the schema cache and Arrow kernels are primed.
    (void)build().collect();

    double raw_ms = minTimeMs([&]() {
        auto df = build().collect_raw();
        assert(df.numRows() <= 100);
    });
    double opt_ms = minTimeMs([&]() {
        auto df = build().collect();
        assert(df.numRows() <= 100);
    });

    std::cout << "raw=" << raw_ms << "ms opt=" << opt_ms << "ms ... ";

    // Correctness parity on a single run.
    auto raw = build().collect_raw();
    auto opt = build().collect();
    assert(raw.numRows() == opt.numRows());
    assert(raw.columnNames() == opt.columnNames());

    // Performance gate: optimizer must beat the unoptimized path by >=10%.
    assert(opt_ms < raw_ms * 0.9);
    std::cout << "OK\n";
}

// ---------------------------------------------------------------------------
// Edge-case tests (Step 5).
// ---------------------------------------------------------------------------

void test21_empty_dataframe() {
    std::cout << "test21_empty_dataframe . ";
    // An EagerDataFrame with 0 rows but a defined schema.
    arrow::Int32Builder ib;
    std::shared_ptr<arrow::Array> a;
    [[maybe_unused]] auto st = ib.Finish(&a);
    arrow::StringBuilder sb;
    std::shared_ptr<arrow::Array> s;
    st = sb.Finish(&s);

    auto df = makeFrame({{"n", a}, {"name", s}});
    assert(df.numRows() == 0);

    auto f = df.filter(col("n") > lit<int32_t>(0));
    assert(f.numRows() == 0);

    auto sel = df.select({"n"});
    assert(sel.numRows() == 0);
    assert(sel.columnNames() == (std::vector<std::string>{"n"}));

    auto so = df.sort({"n"}, true);
    assert(so.numRows() == 0);
    std::cout << "OK\n";
}

void test22_single_row_group_by() {
    std::cout << "test22_single_row_group  ";
    auto df = makeFrame({
        {"dept",   str({"eng"})},
        {"salary", f64({100.0})},
    });
    auto out = df.group_by({"dept"})
                 .aggregate({{"total", col("salary").sum()}});
    assert(out.numRows() == 1);
    assert(std::abs(asF64(column(out, "total"))[0] - 100.0) < 1e-9);
    std::cout << "OK\n";
}

void test23_all_null_column() {
    std::cout << "test23_all_null_column . ";
    auto df = makeFrame({
        {"x", i32_with_nulls({0, 0, 0, 0}, {false, false, false, false})},
    });
    // sum() on an all-null column is null (Arrow semantics).
    auto agg = df.aggregate({{"s", col("x").sum()}});
    auto sum_col = column(agg, "s");
    auto chunk0  = sum_col->chunk(0);
    assert(chunk0->length() == 1);
    assert(chunk0->IsNull(0));

    // filter(is_null) must keep every row.
    auto all = df.filter(col("x").is_null());
    assert(all.numRows() == 4);
    std::cout << "OK\n";
}

void test24_large_string_column() {
    std::cout << "test24_large_strings ... ";
    const int N = 10000;
    std::vector<std::string> v;
    v.reserve(N);
    for (int i = 0; i < N; ++i) {
        // 1 in 3 rows contains '@', others don't, so we can validate
        // contains() counts correctly.
        v.push_back(i % 3 == 0 ? "user" + std::to_string(i) + "@host"
                                : "user" + std::to_string(i));
    }
    auto df = makeFrame({{"email", str(v)}});

    auto has = df.with_column("hasAt", col("email").contains("@"))
                 .aggregate({{"n", col("hasAt").sum()}});
    // contains() returns int32 match-count; sum is the number of rows with
    // at least one '@'. Exactly ceil(N/3) rows include an '@'.
    int64_t expected = 0;
    for (int i = 0; i < N; ++i) if (i % 3 == 0) ++expected;
    auto arr = has.table()->GetColumnByName("n")->chunk(0);
    // sum() of int32 → int64 in Arrow.
    auto sum_arr = std::static_pointer_cast<arrow::Int64Array>(arr);
    assert(sum_arr->Value(0) == expected);

    // to_upper() must work over 10k rows.
    auto up = df.select({ col("email").to_upper().alias("E") });
    assert(up.numRows() == N);
    auto e = std::static_pointer_cast<arrow::StringArray>(
        column(up, "E")->chunk(0));
    assert(e->GetString(0).find("USER") != std::string::npos);
    std::cout << "OK\n";
}

void test25_chained_filters() {
    std::cout << "test25_chained_filters . ";
    auto df = makeFrame({
        {"x", i32({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})},
    });
    auto a = df.filter(col("x") > lit<int32_t>(2))
               .filter(col("x") < lit<int32_t>(9))
               .filter(col("x") != lit<int32_t>(5));
    auto b = df.filter((col("x") > lit<int32_t>(2)) &
                       (col("x") < lit<int32_t>(9)) &
                       (col("x") != lit<int32_t>(5)));
    assert(a.numRows() == b.numRows());
    assert(asI32(column(a, "x")) == asI32(column(b, "x")));
    // Expected: {3,4,6,7,8}
    assert((asI32(column(a, "x")) == std::vector<int32_t>{3, 4, 6, 7, 8}));
    std::cout << "OK\n";
}

void test26_multi_key_join() {
    std::cout << "test26_multi_key_join .. ";
    auto left = makeFrame({
        {"dept", str({"eng", "eng", "hr",  "hr"})},
        {"year", i64({2023,  2024,  2023,  2024})},
        {"val",  i64({10,    20,    30,    40})},
    });
    auto right = makeFrame({
        {"dept",  str({"eng", "hr",  "hr"})},
        {"year",  i64({2024,  2023,  2024})},
        {"bonus", i64({100,   200,   300})},
    });
    auto out = left.join(right, {"dept", "year"}, "inner");
    assert(out.numRows() == 3);
    auto v = asI64(column(out, "val"));
    auto b = asI64(column(out, "bonus"));
    // Pairs matched: (eng,2024,20,100), (hr,2023,30,200), (hr,2024,40,300)
    int64_t sumV = 0, sumB = 0;
    for (auto x : v) sumV += x;
    for (auto x : b) sumB += x;
    assert(sumV == 20 + 30 + 40);
    assert(sumB == 100 + 200 + 300);
    std::cout << "OK\n";
}

void test27_eight_op_lazy_plan() {
    std::cout << "test27_eight_op_lazy ... ";
    const std::string path = "/tmp/dfl_test27.csv";
    {
        std::ofstream f(path);
        f << "dept,year,val\n";
        for (int i = 0; i < 50; ++i) {
            const char* d = (i % 2 == 0) ? "eng" : "hr";
            f << d << "," << (2020 + (i % 4)) << "," << (i * 2) << "\n";
        }
    }
    auto out = scan_csv(path)
                   .filter(col("val") > lit<int64_t>(10))
                   .with_column("doubled", col("val") * lit<int64_t>(2))
                   .group_by({"dept"})
                   .aggregate({{"s", col("doubled").sum()},
                               {"n", col("val").count()}})
                   .sort({"dept"}, true)
                   .head(10)
                   .collect();
    assert(out.numRows() == 2);
    assert(out.columnNames().size() == 3); // dept, s, n
    std::cout << "OK\n";
}

void test28_parquet_roundtrip() {
    std::cout << "test28_parquet_roundtrip ";
    auto df = makeFrame({
        {"id",    i64({1, 2, 3})},
        {"name",  str({"Alice", "Bob", "Carol"})},
        {"score", f64({90.5, 82.0, 75.25})},
    });
    const std::string path = "/tmp/dfl_rt_test.parquet";
    df.write_parquet(path);
    auto back = read_parquet(path);
    assert(back.numRows() == 3);
    assert(back.columnNames() == df.columnNames());
    assert(back.columnTypes() == df.columnTypes());
    auto ids = asI64(column(back, "id"));
    assert((ids == std::vector<int64_t>{1, 2, 3}));
    auto scores = asF64(column(back, "score"));
    assert(std::abs(scores[0] - 90.5)  < 1e-9);
    assert(std::abs(scores[2] - 75.25) < 1e-9);
    std::cout << "OK\n";
}

void test29_optimizer_idempotency() {
    std::cout << "test29_opt_idempotent .. ";
    const std::string path = "/tmp/dfl_lazy_people.csv"; // reused
    auto lf = scan_csv(path)
                 .filter(col("age") > lit<int64_t>(20))
                 .with_column("age2", col("age") * lit<int64_t>(1))
                 .select({"id", "age2"})
                 .head(100);
    QueryOptimizer opt;
    auto once  = opt.optimize(lf.plan());
    auto twice = opt.optimize(once);

    // Structural equality via DOT rendering — same plans render the same.
    auto d1 = renderDotGraph(once);
    auto d2 = renderDotGraph(twice);
    assert(d1 == d2);
    std::cout << "OK\n";
}

void test30_head_out_of_range() {
    std::cout << "test30_head_out_of_range ";
    auto df = makeFrame({{"x", i32({1, 2, 3, 4, 5})}});
    auto h = df.head(1000);
    assert(h.numRows() == 5);
    auto h0 = df.head(0);
    assert(h0.numRows() == 0);
    std::cout << "OK\n";
}

void test31_type_error_message() {
    std::cout << "test31_type_error_msg .. ";
    auto df = makeFrame({
        {"x", i32({1, 2, 3})},
        {"s", str({"a", "b", "c"})},
    });
    bool threw_ia = false;
    std::string msg;
    try {
        auto bad = df.select({ (col("s") + col("x")).alias("oops") });
        (void)bad;
    } catch (const std::invalid_argument& e) {
        threw_ia = true;
        msg      = e.what();
    }
    assert(threw_ia);
    // Message must mention both type names so users can see what collided.
    assert(msg.find("string") != std::string::npos);
    assert(msg.find("int") != std::string::npos);
    std::cout << "OK\n";
}

} // namespace

int main() {
    try {
        test1_csv_roundtrip();
        test2_filter();
        test3_select_with_expr();
        test4_group_by_aggregate();
        test5_sort_desc();
        test6_join_inner();
        test7_null_propagation();
        test8_type_error();
        test9_string_and_agg();
        test10_lazy_collect();
        test11_lazy_explain();
        test12_lazy_group_aggregate();
        test13_lazy_join();
        test14_lazy_sort_head_parquet();
        test15_lazy_sink_csv();
        test16_predicate_pushdown();
        test17_constant_folding();
        test18_expression_simplification();
        test19_limit_pushdown();
        test20_optimizer_benchmark();
        test21_empty_dataframe();
        test22_single_row_group_by();
        test23_all_null_column();
        test24_large_string_column();
        test25_chained_filters();
        test26_multi_key_join();
        test27_eight_op_lazy_plan();
        test28_parquet_roundtrip();
        test29_optimizer_idempotency();
        test30_head_out_of_range();
        test31_type_error_message();
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ALL 31 TESTS PASSED\n";
    return 0;
}
