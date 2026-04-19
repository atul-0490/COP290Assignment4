#include "LazyDataFrame.hpp"

namespace dfl {

LazyDataFrame::LazyDataFrame() : plan_(nullptr) {}

LazyDataFrame::LazyDataFrame(std::shared_ptr<LogicalNode> plan)
    : plan_(std::move(plan)) {}

// --- Operations (stubs) ---

LazyDataFrame LazyDataFrame::select(const std::vector<std::string>& /*columns*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::select(const std::vector<ExprBuilder>& /*exprs*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::filter(const ExprBuilder& /*predicate*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::with_column(const std::string& /*name*/,
                                         const ExprBuilder& /*expr*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::group_by(const std::vector<std::string>& /*keys*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::aggregate(
    const std::map<std::string, ExprBuilder>& /*aggMap*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::join(const LazyDataFrame& /*other*/,
                                  const std::vector<std::string>& /*on*/,
                                  const std::string& /*how*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& /*columns*/,
                                  bool /*ascending*/) const {
    return LazyDataFrame(plan_);
}

LazyDataFrame LazyDataFrame::head(int64_t /*n*/) const {
    return LazyDataFrame(plan_);
}

// --- Sinks / terminals (stubs) ---

void LazyDataFrame::sink_csv(const std::string& /*path*/) const {}
void LazyDataFrame::sink_parquet(const std::string& /*path*/) const {}

EagerDataFrame LazyDataFrame::collect() const {
    return EagerDataFrame();
}

void LazyDataFrame::explain(const std::string& /*pngPath*/) const {}

std::shared_ptr<LogicalNode> LazyDataFrame::plan() const { return plan_; }

} // namespace dfl
