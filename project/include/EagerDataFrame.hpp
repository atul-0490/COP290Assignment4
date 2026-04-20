#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "DataFrame.hpp"
#include "Expr.hpp"

namespace dfl {

/// EagerDataFrame — immediate execution backed by an `arrow::Table`.
///
/// All operations return a brand-new `EagerDataFrame` (immutable-style API).
/// The underlying Arrow table is shared where possible to avoid copying.
class EagerDataFrame : public DataFrame {
public:
    EagerDataFrame();
    explicit EagerDataFrame(std::shared_ptr<arrow::Table> table);

    // --- DataFrame interface ---
    std::vector<std::string> columnNames() const override;
    std::vector<ColType>     columnTypes() const override;
    int64_t                  numRows()    const override;
    void                     print(int64_t maxRows = 20) const override;

    // --- Selection / projection ---
    EagerDataFrame select(const std::vector<std::string>& columns) const;
    EagerDataFrame select(const std::vector<ExprBuilder>& exprs) const;

    /// Initializer-list overload so users can write `df.select({"a","b"})`
    /// unambiguously without the compiler wavering between the string and
    /// ExprBuilder vectors.
    EagerDataFrame select(std::initializer_list<const char*> columns) const;

    // --- Row operations ---
    EagerDataFrame filter(const ExprBuilder& predicate) const;
    EagerDataFrame with_column(const std::string& name,
                               const ExprBuilder& expr) const;

    // --- Grouping / aggregation ---
    EagerDataFrame group_by(const std::vector<std::string>& keys) const;
    EagerDataFrame aggregate(const std::map<std::string, ExprBuilder>& aggMap) const;

    // --- Joins / ordering / limits ---
    EagerDataFrame join(const EagerDataFrame& other,
                        const std::vector<std::string>& on,
                        const std::string& how = "inner") const;
    EagerDataFrame sort(const std::vector<std::string>& columns,
                        bool ascending = true) const;
    EagerDataFrame head(int64_t n) const;

    // --- I/O ---
    void write_csv(const std::string& path) const;
    void write_parquet(const std::string& path) const;

    // --- Access ---
    std::shared_ptr<arrow::Table> table() const;

private:
    std::shared_ptr<arrow::Table> table_;

    /// When non-empty, marks this frame as "grouped" — `aggregate()` will
    /// bucket rows by these key columns before applying each aggregation.
    /// `group_by()` returns a new EagerDataFrame with this set.
    std::vector<std::string> group_keys_;

    /// Evaluate a scalar / array-producing expression against this frame.
    /// Returns a ChunkedArray with one entry per row of the underlying table.
    std::shared_ptr<arrow::ChunkedArray> evalExpr(
        const std::shared_ptr<Expr>& expr) const;
};

} // namespace dfl
