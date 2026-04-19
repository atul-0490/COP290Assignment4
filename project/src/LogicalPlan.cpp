#include "LogicalPlan.hpp"

namespace dfl {

std::string renderDotGraph(const std::shared_ptr<LogicalNode>& /*root*/) {
    return "digraph G {}\n";
}

} // namespace dfl
