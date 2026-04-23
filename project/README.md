# DataFrameLib

A high-performance C++17 DataFrame library built on Apache Arrow, inspired
by Pandas (eager execution) and Polars (lazy execution with a query
optimiser). Submitted for **COP290 – Design Practices, Assignment 4**.

---

## Requirements

- **CMake** ≥ 3.20
- **C++17 compiler** (GCC 11+ or Clang 13+)
- **Apache Arrow** C++ library, with Parquet support
- **Graphviz** (optional — used by `explain()` to render plan diagrams)

### Install dependencies

**Ubuntu / Debian**

```bash
sudo apt install -y build-essential cmake pkg-config \
                    libarrow-dev libparquet-dev \
                    graphviz libgraphviz-dev
```

**macOS (Homebrew)**

```bash
brew install cmake apache-arrow graphviz
```

---

## Build Instructions

From the `project/` directory:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

This produces:

- `libdataframelib.a`       — the static library
- `main`                    — a minimal demo binary
- `tests/test_main`         — the full test suite (31 tests)

## Run Tests

```bash
cd build
./tests/test_main
```

The run prints per-test status and ends with
`ALL N TESTS PASSED` on success.

## Run the Demo

```bash
cd build
./main
```

`main.cpp` exercises every major feature (eager + lazy, joins, groupby,
expressions, optimiser) as a quick smoke demo.

---

## API Quick Reference

### Reading data

```cpp
dfl::read_csv(path)     → EagerDataFrame
dfl::read_parquet(path) → EagerDataFrame
dfl::scan_csv(path)     → LazyDataFrame
dfl::scan_parquet(path) → LazyDataFrame
dfl::from_columns(map)  → EagerDataFrame
```

### EagerDataFrame (immediate execution)

```cpp
select({"a", "b"})                    // project columns
select({col("a")+col("b")})           // project expressions
filter(col("x") > lit(10))            // keep matching rows
with_column("c", col("a")*lit(2))    // add/replace a column
group_by({"k"})                       // declare grouping
aggregate({{"s", col("x").sum()}})    // reduce to one row per group
join(other, {"id"}, "inner")          // equi-join (inner/left/right/outer)
sort({"x"}, ascending=true)           // order rows
head(n)                               // first n rows
write_csv(path) / write_parquet(path) // persist to disk
print(maxRows)                        // pretty-print
```

### LazyDataFrame (deferred execution)

The same operator surface as `EagerDataFrame` but every call extends a
logical plan DAG. Nothing executes until `collect()` is called.

```cpp
collect()        // run with optimizer, return EagerDataFrame
collect_raw()    // run WITHOUT optimizer (benchmarking)
explain(path)    // render plan DAG to a .png via Graphviz
plan()           // inspect the raw plan root
sink_csv(path) / sink_parquet(path)
```

### Expression DSL

```cpp
col("x")                 // column reference
lit(3), lit(3.14), lit("hi"), lit(true)
x + y, x - y, x * y, x / y, x % y
x == y, x != y, x < y, x <= y, x > y, x >= y
x & y   (AND),  x | y   (OR),  ~x   (NOT)
x.abs(), x.is_null(), x.is_not_null(), x.alias("z")
x.length(), x.contains("s"), x.starts_with("s"), x.ends_with("s")
x.to_lower(), x.to_upper()
x.sum(), x.mean(), x.count(), x.min(), x.max()    // aggregations
```

---

## Architecture Overview

DataFrameLib is structured as three layers on top of Apache Arrow.

**EagerDataFrame** is a thin, immutable wrapper around
`std::shared_ptr<arrow::Table>`. Every operation copies the pointer
and runs Arrow's compute kernels directly, so the API feels like
Pandas but data-flow is zero-copy and vectorised.

**LazyDataFrame** builds a logical plan instead of running anything.
Each operation appends a `LogicalNode` (e.g. `FilterNode`, `JoinNode`)
to a `shared_ptr`-based DAG. The root is handed to the
**QueryOptimizer** on `collect()`, which runs five rewrite rules to a
fixed point (max 10 iterations) before the executor walks the DAG
depth-first and materialises the result back into an
`EagerDataFrame`.

The **expression system** is a small AST (`ColExpr`, `LitExpr`,
`BinaryExpr`, `UnaryExpr`, `AggExpr`, `StringExpr`, `AliasExpr`) wrapped
by the operator-overloading `ExprBuilder`. Both execution modes share
the same expression evaluator (`EagerDataFrame::evalExpr`), which maps
onto Arrow Compute. This is what lets the `QueryOptimizer` constant-fold
expressions against a synthetic 1-row table.

---

## Query Optimizations Implemented

1. **Predicate Pushdown** — push `FilterNode`s below `JoinNode` and
   `GroupByNode` when the predicate references only one side / only
   group keys, so fewer rows flow upward.
2. **Projection Pushdown** — propagate the set of required columns
   top-down and annotate `ScanNode::projected_columns` so Parquet /
   CSV reads only the columns actually used.
3. **Constant Folding** — evaluate fully-constant sub-expressions at
   plan time (`lit(3)+lit(4) → lit(7)`).
4. **Expression Simplification** — algebraic identities
   (`x+0 → x`, `x*1 → x`, `~~x → x`, `x AND true → x`, …).
5. **Limit Pushdown** — swap `LimitNode` below row-preserving nodes
   (`SelectNode`, `WithColNode`) and absorb it into
   `ScanNode::row_limit`.

On the included benchmark (scan → join → filter → select → head over
two 20 000-row CSVs) the optimiser produces an end-to-end speedup of
**~2×** vs `collect_raw()`.

---

## Repository Layout

```
project/
├── CMakeLists.txt
├── README.md
├── report.md / report.pdf
├── include/              ← public API headers (Doxygen-documented)
├── src/                  ← library sources
├── tests/test_main.cpp   ← 31 correctness + benchmark tests
├── main.cpp              ← demo
└── prepare_submission.sh ← clean + tar packaging helper
```
