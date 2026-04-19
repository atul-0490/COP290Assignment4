# DataFrameLib

A high-performance C++17 DataFrame library built on Apache Arrow, inspired by
Pandas (eager execution) and Polars (lazy execution with a query optimiser).

This project is the Assignment 4 submission for COP290 – Design Practices.

## Features

- **`EagerDataFrame`** — immediate, Pandas-style execution backed by
  `arrow::Table`.
- **`LazyDataFrame`** — builds a logical plan DAG and defers execution until
  `collect()` is called.
- **`QueryOptimizer`** — rule-based rewrites:
  - Predicate pushdown
  - Projection pushdown
  - Constant folding
  - Expression simplification
  - Limit pushdown
- **Expression DSL** — `col("x") + lit(1)`, comparisons, boolean logic,
  string functions, aggregations, with strict type safety and null semantics.
- **Apache Arrow** I/O for CSV and Parquet.
- **Graphviz / Boost.Graph** DAG rendering via `LazyDataFrame::explain(path)`.

## Requirements

- CMake ≥ 3.20
- A C++17-capable compiler (GCC 9+, Clang 10+)
- Apache Arrow with Parquet support (development headers)
- Either Graphviz (libgvc / libcgraph) **or** Boost.Graph for `explain()`

### Installing dependencies (Ubuntu / Debian)

```bash
sudo apt install -y build-essential cmake pkg-config \
                    libarrow-dev libparquet-dev \
                    libgraphviz-dev
```

## Building

```bash
cd project
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces:

- `libdataframelib.a` — the static library
- `main` — a minimal demo executable
- `tests/test_main` — an API-coverage smoke test

Run the demo:

```bash
./main
```

Run the smoke tests:

```bash
ctest --output-on-failure
```

## Layout

```
project/
├── CMakeLists.txt
├── README.md
├── include/          ← public API headers
├── src/              ← library sources
├── tests/            ← smoke + correctness tests
└── main.cpp          ← demo entry point
```

## Status

This checkpoint is the skeleton build: every public API symbol is declared
and links, but most operations are stubs. Subsequent steps flesh out the
expression evaluator, eager operators, lazy executor, and the query
optimiser.
