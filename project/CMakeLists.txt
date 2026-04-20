cmake_minimum_required(VERSION 3.20)
project(DataFrameLib CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Apache Arrow is always required. Parquet is strongly recommended and is
# treated as required by the assignment spec, but we allow opting out with
# -DDFL_REQUIRE_PARQUET=OFF so the skeleton can be validated on machines
# without libparquet installed.
option(DFL_REQUIRE_PARQUET "Require Parquet support (assignment default)" ON)

find_package(Arrow REQUIRED)
# ArrowCompute holds the scalar/aggregate compute kernels (add, filter, sort,
# ...) on Debian/Ubuntu's Arrow packaging. The Arrow library itself ships
# without kernels registered; linking libarrow_compute pulls them in.
find_package(ArrowCompute QUIET)

if(DFL_REQUIRE_PARQUET)
    find_package(Parquet QUIET)
    if(NOT Parquet_FOUND)
        message(WARNING
            "Parquet not found. Parquet is required by the assignment but is "
            "missing on this machine. Configure with -DDFL_REQUIRE_PARQUET=OFF "
            "to proceed without Parquet for local skeleton validation.")
    endif()
endif()

# Graph rendering: prefer Graphviz's `gvc` library for native rendering; fall
# back to Boost.Graph if Graphviz development headers are unavailable. The
# library itself only *needs* one of them.
set(DFL_GRAPH_BACKEND "")
set(DFL_GRAPH_LIBS "")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GRAPHVIZ QUIET libgvc libcgraph)
endif()

if(GRAPHVIZ_FOUND)
    set(DFL_GRAPH_BACKEND "graphviz")
    set(DFL_GRAPH_LIBS    ${GRAPHVIZ_LIBRARIES})
    message(STATUS "DataFrameLib: using Graphviz for DAG rendering")
else()
    find_package(Boost QUIET COMPONENTS graph)
    if(Boost_FOUND)
        set(DFL_GRAPH_BACKEND "boost")
        set(DFL_GRAPH_LIBS    Boost::graph)
        message(STATUS "DataFrameLib: using Boost.Graph for DAG rendering")
    else()
        message(WARNING
            "DataFrameLib: neither Graphviz nor Boost.Graph found. "
            "explain() will emit DOT text only; rendering to PNG will require "
            "the `dot` CLI at runtime.")
    endif()
endif()

# -----------------------------------------------------------------------------
# Static library
# -----------------------------------------------------------------------------
add_library(dataframelib STATIC
    src/EagerDataFrame.cpp
    src/LazyDataFrame.cpp
    src/QueryOptimizer.cpp
    src/Expr.cpp
    src/LogicalPlan.cpp
    src/IO.cpp
    src/TypeUtils.cpp
)

target_include_directories(dataframelib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Pick whichever Arrow / Parquet targets were exported by the cmake package.
if(TARGET Arrow::arrow_static)
    target_link_libraries(dataframelib PUBLIC Arrow::arrow_static)
else()
    target_link_libraries(dataframelib PUBLIC Arrow::arrow_shared)
endif()

if(TARGET ArrowCompute::arrow_compute_static)
    target_link_libraries(dataframelib PUBLIC ArrowCompute::arrow_compute_static)
elseif(TARGET ArrowCompute::arrow_compute_shared)
    target_link_libraries(dataframelib PUBLIC ArrowCompute::arrow_compute_shared)
endif()

if(TARGET Parquet::parquet_static)
    target_link_libraries(dataframelib PUBLIC Parquet::parquet_static)
    target_compile_definitions(dataframelib PUBLIC DFL_HAVE_PARQUET=1)
elseif(TARGET Parquet::parquet_shared)
    target_link_libraries(dataframelib PUBLIC Parquet::parquet_shared)
    target_compile_definitions(dataframelib PUBLIC DFL_HAVE_PARQUET=1)
endif()

if(DFL_GRAPH_BACKEND STREQUAL "graphviz")
    target_include_directories(dataframelib PUBLIC ${GRAPHVIZ_INCLUDE_DIRS})
    target_link_directories   (dataframelib PUBLIC ${GRAPHVIZ_LIBRARY_DIRS})
    target_link_libraries     (dataframelib PUBLIC ${DFL_GRAPH_LIBS})
    target_compile_definitions(dataframelib PUBLIC DFL_USE_GRAPHVIZ=1)
elseif(DFL_GRAPH_BACKEND STREQUAL "boost")
    target_link_libraries     (dataframelib PUBLIC ${DFL_GRAPH_LIBS})
    target_compile_definitions(dataframelib PUBLIC DFL_USE_BOOST_GRAPH=1)
endif()

# -----------------------------------------------------------------------------
# Main demo executable
# -----------------------------------------------------------------------------
add_executable(main main.cpp)
target_link_libraries(main PRIVATE dataframelib)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
