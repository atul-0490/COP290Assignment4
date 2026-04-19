#include "IO.hpp"

namespace dfl {

EagerDataFrame read_csv(const std::string& /*path*/) {
    return EagerDataFrame();
}

EagerDataFrame read_parquet(const std::string& /*path*/) {
    return EagerDataFrame();
}

LazyDataFrame scan_csv(const std::string& /*path*/) {
    return LazyDataFrame();
}

LazyDataFrame scan_parquet(const std::string& /*path*/) {
    return LazyDataFrame();
}

EagerDataFrame from_columns(
    const std::map<std::string, std::shared_ptr<arrow::Array>>& /*columns*/) {
    return EagerDataFrame();
}

} // namespace dfl
