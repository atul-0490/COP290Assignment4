---
title: "DataFrameLib — Design Document"
subtitle: "COP290 — Assignment 4"
geometry: margin=1in
fontsize: 11pt
---

# DataFrameLib — Design Document

## 1. Architecture Overview

DataFrameLib is organised as three cooperating layers on top of the
Apache Arrow C++ runtime.

- **EagerDataFrame** is the concrete, Pandas-style API. It holds a
  single `std::shared_ptr<arrow::Table>` and implements every operator
  (`filter`, `select`, `with_column`, `group_by`, `aggregate`, `join`,
  `sort`, `head`, I/O) by running Arrow compute kernels directly. The
  API is immutable: every operator returns a new `EagerDataFrame` that
  shares the underlying table wherever possible.

- **LazyDataFrame** exposes the *same* operator surface but defers
  execution. Each operator allocates one `LogicalNode` and returns a
  new `LazyDataFrame` whose plan extends the parent's. The plan is a
  directed acyclic graph of `shared_ptr<LogicalNode>`s — because
  operations are pure, subtrees may be shared across multiple lazy
  frames without issue.

- **QueryOptimizer** takes a plan root and rewrites it to a fixed
  point. It runs each of five rules, normalises the output with a
  structural `planEquals` check, and stops when the plan stops
  changing (or a safety cap of 10 iterations is hit). The rewritten
  plan is then handed to a DFS executor that produces an
  `EagerDataFrame`.

The **expression system** is a small class hierarchy rooted at
`Expr`: `ColExpr`, `LitExpr`, `AliasExpr`, `BinaryExpr`, `UnaryExpr`,
`AggExpr`, `StringExpr`. Users never touch `Expr` directly — they use
the operator-overloading `ExprBuilder` and the top-level `col(name)` /
`lit(value)` functions. Both execution modes share the same evaluator
(`EagerDataFrame::evalExpr`), which lowers each node onto Arrow
Compute. Sharing the evaluator is what makes constant folding trivial:
the optimiser simply evaluates any fully-constant sub-expression
against a synthetic 1-row table.

Two execution modes, one operator surface, one evaluator, one plan
representation — this keeps the library small (< 3000 LOC) while
supporting both imperative and declarative workflows.

## 2. Apache Arrow Integration

Arrow is the physical backbone of every column in the library. Three
properties drove the choice:

- **Columnar layout.** Each column is an `arrow::ChunkedArray` whose
  chunks are contiguous primitive or Utf8 buffers. Vectorised compute
  (sum, equals, greater-than, …) benefits from excellent cache locality
  and auto-vectorisation in the Arrow kernels, which would be
  impractical to reimplement by hand.

- **Zero-copy sharing.** Because an `arrow::Table` is immutable and
  reference-counted, our `EagerDataFrame` copy is an atomic pointer
  copy. `head`, `select`, and `filter` produce new tables but share
  unchanged buffers with their parents.

- **Type safety.** Arrow's `DataType` hierarchy is mapped 1:1 onto our
  `ColType` enum via `arrowTypeToColType`. Unsupported Arrow types are
  rejected at load time — the library will never silently coerce
  `decimal128` or `date32` into a numeric column.

Null handling is inherited verbatim. Every Arrow kernel already
propagates nulls in the way SQL expects (`NULL + 1 = NULL`,
`NULL > 5 = NULL` which is dropped by filter, …) — by composing
kernels we get null-safety "for free".

I/O goes through `arrow::csv::TableReader` and `parquet::arrow::*`.
Reading uses `arrow::io::ReadableFile::Open` whose handle is owned by
the reader object, so closing is automatic on scope exit. Writing is
the symmetric pair (`arrow::csv::MakeCSVWriter`,
`parquet::arrow::WriteTable`).

## 3. Type System

`ColType` (in `TypeUtils.hpp`) is the library's view of a column's
type: `INT32`, `INT64`, `FLOAT32`, `FLOAT64`, `STRING`, `BOOLEAN`.
Every Arrow column is translated to exactly one of these values once
at load time and never changes afterwards.

**Promotion rules** (`promoteTypes`):

- Identical types → same type.
- Integer + integer → widest integer (INT32 + INT64 → INT64).
- Any integer + any float → widest float (mixing a FLOAT32 with an
  INT64 promotes to FLOAT64 for precision).
- Any mix of numeric with STRING or BOOLEAN → `std::invalid_argument`
  whose message includes both colliding type names
  (`"Incompatible types: string and int32"`).

The type check happens inside `evalExpr` for binary operators, which
catches errors at evaluation time (`col("s") + col("i")` throws on
`collect()` / `filter()` / etc.). Unary operators have similar
guards (`NOT` requires boolean, `abs` requires numeric, the string
functions require STRING).

## 4. Lazy Execution and the DAG

A lazy plan is a DAG of `LogicalNode`s with a single root. The node
types are:

| Node          | Role                                         |
|---------------|----------------------------------------------|
| `ScanNode`    | Leaf — reads CSV or Parquet. Carries        |
|               | `projected_columns` and `row_limit`          |
|               | annotations written by the optimiser.        |
| `FilterNode`  | `WHERE predicate`.                          |
| `SelectNode`  | `SELECT expr, expr, …`.                     |
| `WithColNode` | Adds / overwrites one column.                |
| `GroupByNode` | Declares group keys (paired with AggNode).   |
| `AggNode`     | Computes aggregations (per group).           |
| `JoinNode`    | Equi-join; carries `right`, `on`, `how`.     |
| `SortNode`    | Orders rows by `columns`.                    |
| `LimitNode`   | Keeps the first `n` rows.                    |
| `SinkNode`    | Terminal write (CSV or Parquet).             |

`collect()` runs `QueryOptimizer::optimize(plan_)` and then a
recursive DFS executor (`executePlan`) that materialises each node's
children first, then applies the node's operator to its child frames.
The execution path is a straight call into the `EagerDataFrame`
operator of the same name — this is how the library shares
implementation between the two execution modes.

`explain(pngPath)` emits the plan as Graphviz DOT, writes it to a
temp file inside a small RAII guard (`TempDotFile`), and shells out
to `dot -Tpng`. If `dot` is not installed the RAII guard keeps the
`.dot` file and stderr prints its path, so the user can render the
diagram by hand without the call having side-effects beyond one
stray file.

## 5. Query Optimizations

The optimiser is a fixed-point rewriter (`optimize`): it calls each
rule in turn, checks structural plan equality (`planEquals`) against
the previous iteration, and stops when nothing changed (or 10
iterations elapse). Rules cannot loop forever because each one is
individually idempotent once it has fully applied — a property we
verify by test (test 28).

### 5.1 — Constant Folding

**Description.** Any sub-expression composed entirely of `LitExpr`
nodes is evaluated at plan-construction time and replaced with a
single `LitExpr` holding the result.

**Transformation.**
If `e` has no `ColExpr` descendant, rewrite it to `lit(evalExpr(e))`.

**Correctness proof.** `evalExpr` on an expression with no column
references produces the same Arrow scalar as running the kernel at
execution time on any row — column-free expressions are row-
independent. Replacing the expression with that scalar preserves
results.

**Concrete example.**

    Before: filter(col("age") > (lit(20) + lit(10)))
    After:  filter(col("age") > lit(30))

**Performance benefit.** One addition per row eliminated; the
optimised predicate is a single comparison.

### 5.2 — Expression Simplification

**Description.** Rewrite algebraic identities that do not change the
value of an expression.

**Transformation.** Representative rewrites (all null-safe):

| Before                       | After     |
|------------------------------|-----------|
| `x + lit(0)` / `lit(0) + x`  | `x`       |
| `x - lit(0)`                 | `x`       |
| `x * lit(1)` / `lit(1) * x`  | `x`       |
| `x / lit(1)`                 | `x`       |
| `x * lit(0)` / `lit(0) * x`  | `lit(0)`  |
| `x AND lit(true)`            | `x`       |
| `x AND lit(false)`           | `lit(false)` |
| `x OR lit(true)`             | `lit(true)`  |
| `~~x`                        | `x`       |
| `x == x`                     | `lit(true)` |

**Correctness proof.** Each pattern is a well-known ring / Boolean
identity. The null-unsafe rewrites are specifically excluded: we do
*not* rewrite `x - x → lit(0)` because `NULL - NULL` is `NULL`, not
`0`.

**Concrete example.**

    Before: select(col("id") * lit(1), col("x") + lit(0))
    After:  select(col("id"), col("x"))

**Performance benefit.** Fewer compute-kernel invocations. On a
5M-row table each removed identity saves one `Multiply` or `Add`
call.

### 5.3 — Predicate Pushdown

**Description.** Move a `FilterNode` as close to its source as
possible. Two cases are handled: through a `JoinNode` (when the
predicate references only one side) and through a `GroupByNode`
(when the predicate only references grouping keys).

**Transformation.** If `pred` uses only columns of `left` in
`Filter(Join(left, right, on, how), pred)`:

    Before:  Filter(Join(left, right), pred)
    After:   Join(Filter(left, pred), right)

(Analogous for right-side predicates on inner joins; outer joins
only allow pushdown on the preserved side.)

For `Filter(GroupBy(k, child), pred)` where `colRefs(pred) ⊆ k`:

    Before:  Filter(GroupBy(k, child), pred)
    After:   GroupBy(k, Filter(child, pred))

**Correctness proof.** A row is kept by the outer filter iff
`pred` is true for the produced row. For an inner join on `on`,
a row on the left that fails `pred` cannot participate in any
produced row (because `pred` only depends on left columns), so
dropping it below the join changes nothing. For `GroupBy` on keys
`k`: rows in the same group share the same values for `k`, so a
predicate restricted to `k` is constant within a group — either
all rows in the group pass or none do, and moving the test above
or below the grouping is equivalent.

**Concrete example.**

    Before: scan(sales) ⨝ scan(stores)   → filter(store_size > 100)
    After:  scan(sales) ⨝ (scan(stores) → filter(store_size > 100))

**Performance benefit.** The right side of the join shrinks
*before* the join builds its hash table, which for selective
filters can halve or quarter intermediate sizes.

### 5.4 — Projection Pushdown

**Description.** Propagate the set of columns ultimately required at
the sink downward through the plan. At `ScanNode`, write the
intersection (with the file's real schema) into
`ScanNode::projected_columns`, so the reader only materialises
those columns.

**Transformation.** Top-down traversal with a required-set
`R: optional<set<string>>`:

- `Sink / head / sort`: `R` unchanged.
- `SelectNode(exprs)`: replace `R` with `colRefs(exprs)`.
- `FilterNode(pred)`:  `R := R ∪ colRefs(pred)`.
- `WithColNode(name,expr)`: `R := (R \ {name}) ∪ colRefs(expr)`.
- `GroupByNode + AggNode`: `R := keys ∪ colRefs(aggregations)`.
- `JoinNode(on, left, right)`: split `R` into
  `R_left ∪ on` and `R_right ∪ on`, descend into each side.
- `ScanNode`: record `projected_columns = R ∩ file_schema`.

**Correctness proof.** Every node in the plan outputs a subset of
some set of columns. The transformation only ever *drops* columns
that are never referenced upstream of the scan; dropping unused
columns cannot change the result of a relational operator.

**Concrete example.**

    sales.csv has 12 columns. Query only needs (id, amount).
    Before: scan(sales) → select(id, amount)
    After:  scan(sales, projected={id, amount}) → select(id, amount)

**Performance benefit.** For a 12-column Parquet file this reads
2/12 ≈ 17% of the bytes. Even for CSV, converting 10 unused
columns per row to Arrow arrays is skipped.

### 5.5 — Limit Pushdown

**Description.** Move `LimitNode(n)` below every operator that
preserves row count and order. In particular, `LimitNode` is
swapped below `SelectNode` and `WithColNode`, and absorbed into
`ScanNode::row_limit` when it reaches a scan.

**Transformation.**

    LimitNode(n, SelectNode(s,  child))  →  SelectNode(s,  LimitNode(n, child))
    LimitNode(n, WithColNode(w, child))  →  WithColNode(w, LimitNode(n, child))
    LimitNode(n, ScanNode(path,...))     →  ScanNode(path, ..., row_limit=n)

Does *not* push below `FilterNode`, `JoinNode`, `SortNode`,
`GroupByNode`, `AggNode` (these change row count / order).

**Correctness proof.** `SelectNode` and `WithColNode` are per-row
transformations — they neither add, remove, nor reorder rows.
`select(head(x, n)) = head(select(x), n)` because both evaluate
the select on the same set of n rows in the same order. For
`ScanNode`, reading only the first `n` rows of the source file is
exactly what the unoptimised `LimitNode` would produce.

**Concrete example.**

    Before: scan(big.csv) → with_column(c, a+b) → head(5)
    After:  scan(big.csv, row_limit=5) → with_column(c, a+b)

**Performance benefit.** On a 10M-row Parquet file taking `head(5)`
now reads only the first row group. In CSV mode we stop reading
after `n` lines. The `with_column` compute also runs on 5 rows
instead of 10M.

## 6. Memory Management

Every heap allocation in the library is owned by a smart pointer:

- `std::shared_ptr<T>` for values that may have multiple owners:
  `arrow::Table`, `arrow::ChunkedArray`, `LogicalNode`, `Expr`.
- `std::unique_ptr<T>` for exclusively-owned Arrow readers /
  writers inside the I/O code.
- `std::string` / `std::vector` everywhere for owning byte / array
  data — no `char*` escape hatches.

`EagerDataFrame` and `LazyDataFrame` are both "small handle" types:
the only data member is one `shared_ptr`, so copies are
`atomic_fetch_add(1)` on a refcount and moves are a pointer swap.
Both classes have the full defaulted rule-of-5 (`= default`) plus
`noexcept` move operators.

Arrow file handles are never manually closed. Readers returned by
`arrow::io::ReadableFile::Open` are owned by reader objects whose
destructors close them — RAII all the way.

The only filesystem resource we create directly is the intermediate
`.dot` file in `explain()`. This is wrapped in a local
`TempDotFile` struct whose destructor deletes the file unless a
`keep` flag is set (e.g. when `dot` is missing and we want the user
to see the path). `TempDotFile` is non-copyable to prevent
double-delete.

**Exception safety.** Every Arrow API call that returns
`arrow::Status` / `arrow::Result<T>` is checked; failures are
converted to `std::runtime_error(status.ToString())`. Input
validation throws `std::invalid_argument` with a descriptive
message (missing column name, unknown join mode, non-boolean
predicate, …). Because all owning state is held by smart pointers
and standard containers, unwinding is leak-free.

The test suite is run under `-fsanitize=address,undefined` with
`detect_leaks=1`; the complete 31-test run reports zero leaks and
zero UBSan errors.

## 7. Testing

The test binary `tests/test_main` contains 31 tests divided into
five categories:

1. **Schema & eager operators** (tests 1–10): `read_csv`,
   `from_columns`, `select`, `filter`, `with_column`, `group_by +
   aggregate`, `join`, `sort`, `head`, `write_csv`.
2. **Expression & type system** (tests 11–13): arithmetic
   promotion, null handling, string functions, incompatible-type
   detection.
3. **Lazy execution** (tests 14–15): plan construction,
   `collect()`, `explain()`, parity between eager and lazy
   pipelines.
4. **Optimiser** (tests 16–20): one test per rule plus a
   benchmark. Each optimiser test asserts both *structural*
   correctness (inspect the optimised plan) and *semantic*
   correctness (`collect() == collect_raw()`).
5. **Edge cases** (tests 21–31): empty frames, single-row
   aggregations, all-null columns, 10 000-row string columns,
   chained filters, multi-key joins, 8-operation lazy plans,
   parquet round-trip, optimiser idempotency, out-of-range
   `head`, and `std::invalid_argument` on string+int arithmetic.

### Benchmark methodology

The optimiser benchmark (test 20) builds two 20 000-row CSV files
and a four-operator pipeline (`scan → join → filter → select →
head`). It measures `collect_raw()` (no optimiser) vs `collect()`
(optimiser on) using a `minTimeMs` helper that runs 3 warm-up
iterations followed by 5 timed iterations and reports the minimum
— the minimum is a much more stable metric than the mean for
short measurements. The test asserts the optimised run is at least
10 % faster; in practice the speedup is ~2×.

Memory safety is verified by recompiling the suite with
`-fsanitize=address,undefined` and `ASAN_OPTIONS=detect_leaks=1`,
which reports no errors and no leaks across all 31 tests.
