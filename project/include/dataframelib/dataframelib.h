#pragma once

#include <stdexcept>

#include <arrow/status.h>

#include "EagerDataFrame.hpp"
#include "Expr.hpp"
#include "IO.hpp"
#include "LazyDataFrame.hpp"

#ifndef ARROW_THROW_NOT_OK
#define ARROW_THROW_NOT_OK(status_expr)                                            \
	do {                                                                            \
		::arrow::Status _dfl_status = (status_expr);                                \
		if (!_dfl_status.ok()) {                                                    \
			throw std::runtime_error(_dfl_status.ToString());                       \
		}                                                                           \
	} while (0)
#endif

namespace dataframelib = dfl;
