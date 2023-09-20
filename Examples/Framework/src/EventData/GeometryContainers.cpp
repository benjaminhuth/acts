#include "ActsExamples/EventData/GeometryContainers.hpp"

#include "ActsExamples/EventData/IndexSourceLink.hpp"

Acts::GeometryIdentifier ActsExamples::detail::GeometryIdGetter::operator()(
    const ActsExamples::Measurement& meas) const {
  auto f = [](const auto& m) {
    return m.sourceLink().template get<IndexSourceLink>().geometryId();
  };
  return std::visit(f, meas);
}
