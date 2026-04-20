#include <iostream>
#include <memory>
#include <vector>

#include <arrow/api.h>
#include <arrow/builder.h>

#include "EagerDataFrame.hpp"
#include "Expr.hpp"

using dfl::EagerDataFrame;
using dfl::col;
using dfl::lit;

namespace {

// Helper: build an Arrow table with { id: int32, name: string, score: float64 }
// using small in-memory data. Fails fast via abort() on builder errors — this
// is demo code, not a test harness.
std::shared_ptr<arrow::Table> makeDemoTable() {
    arrow::Int32Builder  id_builder;
    arrow::StringBuilder name_builder;
    arrow::DoubleBuilder score_builder;

    const std::vector<int32_t>     ids    = {1, 2, 3, 4, 5};
    const std::vector<std::string> names  = {"Alice", "Bob", "Carol", "Dan", "Eve"};
    const std::vector<double>      scores = {90.5, 82.0, 75.25, 88.0, 95.75};

    if (!id_builder.AppendValues(ids).ok())                  std::abort();
    if (!name_builder.AppendValues(names).ok())              std::abort();
    if (!score_builder.AppendValues(scores).ok())            std::abort();

    std::shared_ptr<arrow::Array> id_arr, name_arr, score_arr;
    if (!id_builder.Finish(&id_arr).ok())       std::abort();
    if (!name_builder.Finish(&name_arr).ok())   std::abort();
    if (!score_builder.Finish(&score_arr).ok()) std::abort();

    auto schema = arrow::schema({
        arrow::field("id",    arrow::int32()),
        arrow::field("name",  arrow::utf8()),
        arrow::field("score", arrow::float64()),
    });

    return arrow::Table::Make(schema, {id_arr, name_arr, score_arr});
}

} // namespace

int main() {
    auto table = makeDemoTable();
    EagerDataFrame df(table);

    std::cout << "-- full frame --\n";
    df.print();

    std::cout << "\n-- filter id > 2, then add double_score column --\n";
    auto out = df.filter(col("id") > lit<int32_t>(2))
                 .with_column("double_score", col("score") * lit<double>(2.0))
                 .select({"id", "name", "score", "double_score"});
    out.print();

    std::cout << "\nBuild OK\n";
    return 0;
}
