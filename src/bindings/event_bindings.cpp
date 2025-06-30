#include "events.hpp"
#include "nbh.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_event_ext(nb::module_ &m) {
  nb::enum_<EventType>(m, "EventType", nb::is_arithmetic())
      .value("MAPPER", EventType::MAPPER)
      .value("RESERVER", EventType::RESERVER)
      .value("LAUNCHER", EventType::LAUNCHER)
      .value("EVICTOR", EventType::EVICTOR)
      .value("COMPUTE_COMPLETER", EventType::COMPUTE_COMPLETER)
      .value("DATA_COMPLETER", EventType::DATA_COMPLETER)
      .value("EVICTOR_COMPLETER", EventType::EVICTOR_COMPLETER)
      .export_values();
}