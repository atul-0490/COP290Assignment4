# Tests subproject — kept minimal for step 1. We do NOT pull in a test
# framework yet; we just build a smoke-test executable that calls every
# public API surface so the linker proves everything resolves.

add_executable(test_main test_main.cpp)
target_link_libraries(test_main PRIVATE dataframelib)

add_test(NAME smoke COMMAND test_main)
