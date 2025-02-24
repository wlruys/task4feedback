#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "events.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_event_ext(nb::module_& m) {
    nb::enum_<EventType>(m, "EventType", nb::is_arithmetic())
        .value("MAPPER", EventType::MAPPER)
        .value("RESERVER", EventType::RESERVER)
        .value("LAUNCHER", EventType::LAUNCHER)
        .value("EVICTOR", EventType::EVICTOR)
        .value("COMPLETER", EventType::COMPLETER)
        .export_values();
}