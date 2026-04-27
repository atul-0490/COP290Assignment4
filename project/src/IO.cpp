#include "IO.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>

#ifdef DFL_HAVE_PARQUET
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#endif

#include "LogicalPlan.hpp"

namespace dfl {

namespace {

void ensureOk(const arrow::Status& s, const std::string& context) {
    if (!s.ok()) {
        throw std::runtime_error(context + ": " + s.ToString());
    }
}

template <typename T>
T unwrap(arrow::Result<T> res, const std::string& context) {
    if (!res.ok()) {
        throw std::runtime_error(context + ": " + res.status().ToString());
    }
    return std::move(res).ValueOrDie();
}

EagerDataFrame from_columns_ordered(
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& columns) {
    if (columns.empty()) return EagerDataFrame();

    int64_t nrows = -1;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> cols;
    fields.reserve(columns.size());
    cols.reserve(columns.size());

    for (const auto& [name, arr] : columns) {
        if (!arr) {
            throw std::runtime_error("from_columns: null array for '" + name + "'");
        }
        if (nrows < 0) nrows = arr->length();
        else if (arr->length() != nrows) {
            throw std::runtime_error("from_columns: column length mismatch");
        }
        fields.push_back(arrow::field(name, arr->type()));
        cols.push_back(std::make_shared<arrow::ChunkedArray>(arr));
    }

    auto schema = arrow::schema(fields);
    return EagerDataFrame(arrow::Table::Make(schema, cols, nrows));
}

} 


EagerDataFrame read_csv(const std::string& path) {
    auto pool = arrow::default_memory_pool();

    auto infile = unwrap(arrow::io::ReadableFile::Open(path, pool),
                         "read_csv: open '" + path + "'");

    auto read_opts    = arrow::csv::ReadOptions::Defaults();
    auto parse_opts   = arrow::csv::ParseOptions::Defaults();
    auto convert_opts = arrow::csv::ConvertOptions::Defaults();

    auto reader = unwrap(
        arrow::csv::TableReader::Make(
            arrow::io::IOContext(pool), infile,
            read_opts, parse_opts, convert_opts),
        "read_csv: make reader");

    auto table = unwrap(reader->Read(), "read_csv: read table");
    return EagerDataFrame(table);
}


EagerDataFrame read_parquet(const std::string& path) {
#ifdef DFL_HAVE_PARQUET
    auto pool   = arrow::default_memory_pool();
    auto infile = unwrap(arrow::io::ReadableFile::Open(path, pool),
                         "read_parquet: open '" + path + "'");

    auto reader = unwrap(parquet::arrow::OpenFile(infile, pool),
                         "read_parquet: open file reader");

    std::shared_ptr<arrow::Table> table;
    ensureOk(reader->ReadTable(&table), "read_parquet: read table");
    return EagerDataFrame(table);
#else
    (void)path;
    throw std::runtime_error(
        "read_parquet: DataFrameLib was built without Parquet support");
#endif
}


LazyDataFrame scan_csv(const std::string& path) {
    auto n       = std::make_shared<ScanNode>();
    n->path      = path;
    n->isParquet = false;
    return LazyDataFrame(n);
}

LazyDataFrame scan_parquet(const std::string& path) {
    auto n       = std::make_shared<ScanNode>();
    n->path      = path;
    n->isParquet = true;
    return LazyDataFrame(n);
}


EagerDataFrame from_columns(
    const std::map<std::string, std::shared_ptr<arrow::Array>>& columns) {
    std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> ordered;
    ordered.reserve(columns.size());
    for (const auto& kv : columns) ordered.push_back(kv);
    return from_columns_ordered(ordered);
}

EagerDataFrame from_columns(
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& columns) {
    return from_columns_ordered(columns);
}

EagerDataFrame from_columns(
    std::initializer_list<std::pair<std::string, std::shared_ptr<arrow::Array>>> columns) {
    std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>> ordered(columns);
    return from_columns_ordered(ordered);
}

} 
