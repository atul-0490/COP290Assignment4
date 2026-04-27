#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <initializer_list>

#include <arrow/api.h>

#include "EagerDataFrame.hpp"
#include "LazyDataFrame.hpp"

namespace dfl {

EagerDataFrame read_csv(const std::string& path);

EagerDataFrame read_parquet(const std::string& path);


LazyDataFrame scan_csv(const std::string& path);

LazyDataFrame scan_parquet(const std::string& path);


EagerDataFrame from_columns( const std::map<std::string, std::shared_ptr<arrow::Array>>& columns);

EagerDataFrame from_columns( const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& columns);

EagerDataFrame from_columns(  std::initializer_list<std::pair<std::string, std::shared_ptr<arrow::Array>>> columns);
} 
