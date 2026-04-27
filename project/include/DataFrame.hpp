#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "TypeUtils.hpp"



namespace dfl
{
class DataFrame {
public:
    virtual ~DataFrame() = default;
    virtual std::vector<std::string> columnNames() const = 0;
    virtual std::vector<ColType> columnTypes() const = 0;
    virtual int64_t numRows() const = 0;
    virtual void print(int64_t maxRows = 20) const = 0;
};

} 
