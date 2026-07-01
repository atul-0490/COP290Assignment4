# DataFrameLib

A practical C++17 DataFrame library built on Apache Arrow with both eager (immediate)
and lazy (optimized) execution modes. Includes a query optimizer with five rewrite rules.

**COP290 – Design Practices, Assignment 4**

---

## Getting started

### Prerequisites

- **CMake** ≥ 3.20
- **C++17 compiler** (GCC 11+ or Clang 13+)
- **Apache Arrow** C++ development headers (with Parquet)
- **Graphviz** (optional, for `explain()` visualization)

### Install dependencies

**Ubuntu / Debian:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config \
  libarrow-dev libparquet-dev graphviz libgraphviz-dev
```

**macOS (Homebrew):**

```bash
brew install cmake apache-arrow graphviz
```

### Build the library

From the `project/` directory:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

Output:
- `build/student_build/libdataframelib.a` — static library
- `build/main` — demo executable
- `build/tests/test_main` — test suite (32 tests)

### Run Tests

```bash
cd build
./tests/test_main
```

All 32 tests validate correctness, null handling, optimizations, and edge cases.
Output ends with `ALL 32 TESTS PASSED` on success.

### Run the Demo

```bash
cd build
./main
```

Demo exercises both eager and lazy modes, joins, groupby, expressions, and the optimizer.

---

## Usage examples

### Eager execution (immediate)

```cpp
#include "dataframelib/dataframelib.h"
using namespace dfl;

// Load CSV
auto df = read_csv("data.csv");

// Select columns
auto subset = df.select({"id", "name", "salary"});

// Filter
auto high_earners = subset.filter(col("salary") > lit<double>(50000.0));

// Add computed column
auto with_tax = high_earners.with_column("tax", 
                                        col("salary") * lit<double>(0.2));

// Write result
with_tax.write_csv("output.csv");
```

### Lazy execution (optimized)

```cpp
// Same operators, but deferred execution
auto result = scan_csv("sales.csv")
  .filter(col("amount") > lit<int64_t>(1000))
  .with_column("commission", col("amount") * lit<double>(0.05))
  .group_by({"salesperson"})
  .aggregate({{"total", col("amount").sum()},
              {"count", col("amount").count()}})
  .sort({"total"}, false)  // descending
  .head(10)
  .collect();  // optimizer runs here, then executes

result.print(10);
```

### Expressions

```cpp
// Binary operations
col("x") + col("y");
col("price") * lit<double>(1.1);  // 10% markup

// Comparisons
col("age") > lit<int32_t>(18);
col("status") == lit("active");

// String functions
col("name").to_upper();
col("email").contains("@");

// Aggregations
col("salary").sum();
col("score").mean();
col("id").count();
```

---

## Architecture

**Three-layer design:**

1. **EagerDataFrame** — thin wrapper around `arrow::Table`. All operators 
   call Arrow compute kernels and return immediately. Immutable, zero-copy 
   semantics.

2. **LazyDataFrame** — builds a logical plan DAG. Every operator appends 
   a `LogicalNode` and returns a new lazy frame. Nothing executes until 
   `collect()` is called.

3. **QueryOptimizer** — rewrites the logical plan using five rules applied 
   to a fixed point, then a DFS executor walks the optimized DAG and 
   materializes the result via eager operators.

**Expression system:**
- Small AST: `ColExpr`, `LitExpr`, `BinaryExpr`, `UnaryExpr`, `AggExpr`, 
  `StringExpr`, `AliasExpr`
- User-facing: operator overloads on `ExprBuilder`, `col()` and `lit()` helpers
- Shared evaluator for both eager and lazy modes (enables constant folding)

---

## Optimizations

1. **Predicate Pushdown** — move filters below joins/groupby when safe
2. **Projection Pushdown** — only read columns actually used
3. **Constant Folding** — evaluate `lit(3) + lit(4)` → `lit(7)` at plan time
4. **Expression Simplification** — apply identities like `x + 0 → x`
5. **Limit Pushdown** — absorb `head(n)` into scans

Typical speedup: **1.5–3×** depending on data and predicate selectivity.

---

## File structure

```
project/
├── include/
│   ├── DataFrame.hpp          (base interface)
│   ├── EagerDataFrame.hpp     (eager execution)
│   ├── LazyDataFrame.hpp      (lazy execution)
│   ├── Expr.hpp               (expression AST)
│   ├── LogicalPlan.hpp        (plan nodes)
│   ├── QueryOptimizer.hpp     (rewrite rules)
│   ├── IO.hpp                 (read/write)
│   ├── TypeUtils.hpp          (type system)
│   └── dataframelib/dataframelib.h  (public header)
├── src/
│   ├── EagerDataFrame.cpp
│   ├── LazyDataFrame.cpp
│   ├── Expr.cpp
│   ├── LogicalPlan.cpp
│   ├── QueryOptimizer.cpp
│   ├── IO.cpp
│   ├── TypeUtils.cpp
│   └── internal/              (utility internals)
├── tests/
│   └── test_main.cpp          (32 tests)
├── main.cpp                   (demo)
└── CMakeLists.txt
```
