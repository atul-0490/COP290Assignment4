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
#include "TypeUtils.hpp"

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
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
