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

/// @brief Immediate-execution DataFrame backed by an `arrow::Table`.
///
/// All operations return a brand-new `EagerDataFrame` (immutable-style
/// API). The underlying Arrow table is shared where possible so most
/// transformations are effectively zero-copy. Missing values are carried
/// through as Arrow nulls and propagated by every compute kernel.
///
/// @code
///   auto df = dfl::read_csv("people.csv");
///   auto adults = df.filter(col("age") >= lit(18))
///                   .select({"name", "age"});
///   adults.print();
/// @endcode
class EagerDataFrame : public DataFrame {
public:
    /// @brief Construct an empty EagerDataFrame with no columns and 0 rows.
    EagerDataFrame();

    /// @brief Wrap an existing Arrow table (shared_ptr semantics).
    /// @param table Non-null `arrow::Table` to take ownership of.
    explicit EagerDataFrame(std::shared_ptr<arrow::Table> table);

    // Rule of 5: enable efficient moves (pointer-only transfer of the
    // underlying Arrow table) and explicit copies. No raw resources are
    // owned beyond shared_ptrs, so the defaulted special members are safe.
    EagerDataFrame(const EagerDataFrame&)                = default;
    EagerDataFrame(EagerDataFrame&&) noexcept            = default;
    EagerDataFrame& operator=(const EagerDataFrame&)     = default;
    EagerDataFrame& operator=(EagerDataFrame&&) noexcept = default;
    ~EagerDataFrame() override                           = default;

    // --- DataFrame interface ---

    /// @copydoc DataFrame::columnNames
    std::vector<std::string> columnNames() const override;
    /// @copydoc DataFrame::columnTypes
    std::vector<ColType>     columnTypes() const override;
    /// @copydoc DataFrame::numRows
    int64_t                  numRows()    const override;
    /// @copydoc DataFrame::print
    void                     print(int64_t maxRows = 20) const override;

    // --- Selection / projection ---

    /// @brief Project a subset of columns by name (preserves order).
    /// @param columns Names of columns to keep.
    /// @return A new frame containing only the listed columns.
    /// @throws std::runtime_error if any name is missing from the schema.
    ///
    /// Example: `df.select({"name", "age"})`.
    EagerDataFrame select(const std::vector<std::string>& columns) const;

    /// @brief Project a set of (possibly computed) columns given as
    ///        `ExprBuilder`s. This is the most flexible form of select.
    /// @param exprs Expressions to evaluate; use `.alias(name)` to rename.
    /// @return A new frame whose columns are the results of `exprs`.
    /// @throws std::runtime_error on missing referenced columns or type errors.
    EagerDataFrame select(const std::vector<ExprBuilder>& exprs) const;

    /// @brief Convenience overload: `df.select({"a","b"})`.
    ///
    /// Disambiguates the braced-init between the string vector and the
    /// ExprBuilder vector overloads.
    /// @param columns Initializer list of string literals naming columns.
    /// @return A new frame with just those columns.
    EagerDataFrame select(std::initializer_list<const char*> columns) const;

    // --- Row operations ---

    /// @brief Keep only rows for which `predicate` evaluates to `true`.
    ///
    /// Null or `false` rows are dropped (Kleene semantics). The predicate
    /// must resolve to a boolean column.
    ///
    /// @param predicate  An ExprBuilder producing a boolean column.
    ///                   Example: `col("age") > lit(30)`.
    /// @return A new frame containing only the passing rows.
    /// @throws std::invalid_argument if the predicate is not boolean.
    /// @throws std::runtime_error    if a referenced column is missing.
    ///
    /// Example: `auto adults = df.filter(col("age") >= lit(18));`
    EagerDataFrame filter(const ExprBuilder& predicate) const;

    /// @brief Add (or overwrite) a column named `name` with the result
    ///        of evaluating `expr`.
    /// @param name Column name to write.
    /// @param expr Expression whose per-row result becomes the new column.
    /// @return     A new frame with the new/overwritten column appended.
    /// @throws     std::runtime_error on type errors or missing columns.
    EagerDataFrame with_column(const std::string& name,
                               const ExprBuilder& expr) const;

    // --- Grouping / aggregation ---

    /// @brief Declare a grouping for the frame. On its own this is
    ///        side-effect-free — the grouping becomes active once
    ///        `aggregate()` is called.
    /// @param keys Ordered list of column names to group by.
    /// @return A new frame that remembers `keys` as its grouping.
    EagerDataFrame group_by(const std::vector<std::string>& keys) const;

    /// @brief Reduce the frame using the aggregations in `aggMap`.
    ///
    /// Each map entry is `{output_column_name, agg_expression}` where
    /// `agg_expression` must be one of `col("x").sum()`, `.mean()`,
    /// `.count()`, `.min()`, `.max()`. If a prior `group_by()` was
    /// applied, one output row per group is produced; otherwise the
    /// frame is reduced to a single row.
    ///
    /// @param aggMap Mapping of output name → aggregation expression.
    /// @return A new frame holding the aggregated results.
    /// @throws std::invalid_argument for non-aggregation expressions.
    EagerDataFrame aggregate(const std::map<std::string, ExprBuilder>& aggMap) const;

    // --- Joins / ordering / limits ---

    /// @brief Equi-join `*this` with `other` on `on` columns.
    ///
    /// @param other Right-hand frame to join against.
    /// @param on    Column names that must exist in BOTH frames.
    /// @param how   One of `"inner"`, `"left"`, `"right"`, `"outer"`.
    /// @return A new frame with the joined rows.
    /// @throws std::invalid_argument for an unknown `how` or mismatched
    ///         key columns.
    EagerDataFrame join(const EagerDataFrame& other,
                        const std::vector<std::string>& on,
                        const std::string& how = "inner") const;

    /// @brief Sort rows by `columns`.
    /// @param columns   Sort keys in priority order.
    /// @param ascending `true` for ascending, `false` for descending.
    /// @return A new sorted frame.
    EagerDataFrame sort(const std::vector<std::string>& columns,
                        bool ascending = true) const;

    /// @brief Keep the first `n` rows. Safe if `n > numRows()`.
    /// @param n  Maximum number of rows to keep (negatives treated as 0).
    /// @return   A new frame with `min(n, numRows())` rows.
    EagerDataFrame head(int64_t n) const;

    // --- I/O ---

    /// @brief Write the frame to `path` in CSV format.
    /// @param path Destination filesystem path.
    /// @throws std::runtime_error on any I/O error.
    void write_csv(const std::string& path) const;

    /// @brief Write the frame to `path` in Parquet format.
    /// @param path Destination filesystem path.
    /// @throws std::runtime_error on any I/O / encoding error.
    void write_parquet(const std::string& path) const;

    // --- Access ---

    /// @brief Read-only access to the underlying Arrow table.
    /// @return A `shared_ptr` to the table (may be null for default ctor).
    std::shared_ptr<arrow::Table> table() const;

    /// @brief Evaluate an expression against this frame, returning the
    ///        resulting column array.
    ///
    /// Exposed (intentionally public) so the QueryOptimizer can reuse
    /// this evaluator when constant-folding pure expressions against a
    /// synthetic 1-row table.
    ///
    /// @param expr Non-null expression tree.
    /// @return     A `ChunkedArray` holding the per-row result.
    /// @throws     std::runtime_error on missing columns / type errors.
    std::shared_ptr<arrow::ChunkedArray> evalExpr(
        const std::shared_ptr<Expr>& expr) const;

private:
    std::shared_ptr<arrow::Table> table_;

    /// When non-empty, marks this frame as "grouped" — `aggregate()` will
    /// bucket rows by these key columns before applying each aggregation.
    /// `group_by()` returns a new EagerDataFrame with this set.
    std::vector<std::string> group_keys_;
};

} // namespace dfl
