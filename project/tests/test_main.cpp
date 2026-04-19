// Smoke tests — step 1 only checks that every public API symbol links.
// Real correctness tests land in later steps.

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DataFrame.hpp"
#include "EagerDataFrame.hpp"
#include "Expr.hpp"
#include "IO.hpp"
#include "LazyDataFrame.hpp"
#include "LogicalPlan.hpp"
#include "QueryOptimizer.hpp"
#include "TypeUtils.hpp"

using namespace dfl;

namespace {

void touchTypeUtils() {
    (void)promoteTypes(ColType::INT32, ColType::INT64);
    (void)isNumeric(ColType::FLOAT64);
    (void)colTypeToString(ColType::STRING);
}

void touchExprs() {
    auto e = (col("a") + lit<int32_t>(1)) * col("b")
                 .alias("scaled")
                 .abs();
    (void)e;

    auto s = col("name").to_lower().starts_with("x");
    (void)s;

    auto agg = col("x").sum();
    (void)agg;

    auto pred = (col("a") > lit<int32_t>(0)) & (col("b") < lit<int32_t>(5));
    (void)pred;

    auto notp = ~col("flag");
    (void)notp;
}

void touchEager() {
    EagerDataFrame df;
    (void)df.columnNames();
    (void)df.columnTypes();
    (void)df.numRows();
    df.print();

    (void)df.select(std::vector<std::string>{"a"});
    (void)df.select(std::vector<ExprBuilder>{col("a")});
    (void)df.filter(col("a") > lit<int32_t>(0));
    (void)df.with_column("z", col("a") + col("b"));
    (void)df.group_by({"dept"});
    (void)df.aggregate(std::map<std::string, ExprBuilder>{{"s", col("x").sum()}});
    (void)df.join(df, {"id"}, "inner");
    (void)df.sort({"a"}, true);
    (void)df.head(5);

    df.write_csv("/tmp/dfl_nowhere.csv");
    df.write_parquet("/tmp/dfl_nowhere.parquet");
}

void touchLazy() {
    LazyDataFrame lf;
    (void)lf.select(std::vector<std::string>{"a"});
    (void)lf.select(std::vector<ExprBuilder>{col("a")});
    (void)lf.filter(col("a") > lit<int32_t>(0));
    (void)lf.with_column("z", col("a") + col("b"));
    (void)lf.group_by({"dept"});
    (void)lf.aggregate(std::map<std::string, ExprBuilder>{{"s", col("x").sum()}});
    (void)lf.join(lf, {"id"}, "inner");
    (void)lf.sort({"a"}, true);
    (void)lf.head(5);

    lf.sink_csv("/tmp/dfl_nowhere.csv");
    lf.sink_parquet("/tmp/dfl_nowhere.parquet");
    (void)lf.collect();
    lf.explain("/tmp/dfl_plan.png");
}

void touchIO() {
    (void)read_csv("/tmp/nope.csv");
    (void)read_parquet("/tmp/nope.parquet");
    (void)scan_csv("/tmp/nope.csv");
    (void)scan_parquet("/tmp/nope.parquet");
    (void)from_columns({});
}

void touchOptimizer() {
    QueryOptimizer opt;
    (void)opt.optimize(nullptr);
}

void touchDotRender() {
    (void)renderDotGraph(nullptr);
}

} // namespace

int main() {
    touchTypeUtils();
    touchExprs();
    touchEager();
    touchLazy();
    touchIO();
    touchOptimizer();
    touchDotRender();

    std::cout << "Smoke tests linked OK\n";
    return 0;
}
