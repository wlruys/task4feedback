
#include "devices.hpp"
#include "nbh.hpp"
#include "resources.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void init_device_ext(nb::module_ &m) {

  m.attr("BYTES_IN_POWER") = BYTES_IN_POWER;
  m.attr("MAX_VCUS") = MAX_VCUS;
  m.attr("MAX_TIME") = MAX_TIME;
  m.attr("MAX_COPIES") = MAX_COPIES;
  m.attr("num_resource_types") = num_resource_types;

  nb::enum_<ResourceType>(m, "ResourceType", nb::is_arithmetic())
      .value("VCUS", ResourceType::VCUS)
      .value("MEM", ResourceType::MEM)
      .value("TIME", ResourceType::TIME)
      .export_values();

  nb::enum_<DeviceType>(m, "DeviceType", nb::is_arithmetic())
      .value("NONE", DeviceType::NONE)
      .value("CPU", DeviceType::CPU)
      .value("GPU", DeviceType::GPU)
      .export_values();

  nb::class_<Resources>(m, "Resources")
      .def(nb::init<vcu_t, mem_t>(), "vcus"_a, "mem"_a)
      .def_ro("vcus", &Resources::vcu)
      .def_ro("mem", &Resources::mem);

  using ResourceEventType = ResourceEvent<int64_t>;
  nb::class_<ResourceEventType>(m, "ResourceEvent")
      .def(nb::init<>())
      .def(nb::init<timecount_t, mem_t>())
      .def_ro("time", &ResourceEventType::time)
      .def_ro("resource", &ResourceEventType::resource);

  using ResourceArrayType = ResourceEventArray<int64_t>;
  nb::class_<ResourceArrayType>(m, "ResourceEventVector")
      .def(nb::init<>())
      .def_rw("size", &ResourceArrayType::size)
      .def_rw("times", &ResourceArrayType::times)
      .def_rw("resources", &ResourceArrayType::resources);

  nb::class_<Device>(m, "Device")
      .def(nb::init<>())
      .def(nb::init<devid_t, DeviceType, vcu_t, mem_t>(), "id"_a, "arch"_a, "vcu"_a, "mem"_a)
      .def_ro("id", &Device::id)
      .def_ro("arch", &Device::arch)
      .def_ro("max_resources", &Device::max_resources, nb::rv_policy::copy)
      .def("get_mem", &Device::get_mem)
      .def("get_vcu", &Device::get_vcu);

  nb::class_<Devices>(m, "Devices")
      .def(nb::init<>())
      .def(nb::init<std::size_t>(), "n_devices"_a)
      .def("create_device", &Devices::create_device, "id"_a, "name"_a, "arch"_a, "vcus"_a, "mem"_a)
      .def("append_device", &Devices::append_device, "name"_a, "arch"_a, "vcus"_a, "mem"_a)
      .def("get_device", nb::overload_cast<devid_t>(&Devices::get_device, nb::const_), "id"_a,
           nb::rv_policy::reference_internal)
      .def("get_device_id", &Devices::get_device_id, "name"_a)
      .def("get_local_id", &Devices::get_local_id, "global_id"_a)
      .def("get_global_id", &Devices::get_global_id, "arch"_a, "local_id"_a)
      .def("get_type", &Devices::get_type, "id"_a)
      .def("size", &Devices::size)
      .def("get_name", nb::overload_cast<devid_t>(&Devices::get_name, nb::const_), "id"_a,
           nb::rv_policy::copy);
}