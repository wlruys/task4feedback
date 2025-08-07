#include "nbh.hpp"
#include "parmetis_wrapper.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>

namespace nb = nanobind;
using namespace nb::literals;

void init_parmetis_ext(nb::module_ &m) {
  nb::class_<ParMETIS_wrapper>(m, "ParMETIS_wrapper")
      .def(nb::init<>())
      .def(nb::init<int>())
      .def("print_info", &ParMETIS_wrapper::print_info)
      // .def("callParMETIS", &ParMETIS_wrapper::callParMETIS, "vtxdist"_a, "xadj"_a, "adjncy"_a,
      //      "vwgt"_a, "vsize"_a, "adjwgt"_a, "wgtflag"_a, "numflag"_a, "ncon"_a, "tpwgts"_a,
      //      "ubvec"_a, "itr"_a, "part"_a);
      .def(
          "callParMETIS",
          [](ParMETIS_wrapper &self, nb::ndarray<int32_t> vtxdist, nb::ndarray<int32_t> xadj,
             nb::ndarray<int32_t> adjncy, nb::ndarray<int32_t> vwgt, nb::ndarray<int32_t> vsize,
             nb::ndarray<int32_t> adjwgt, int32_t wgtflag, int32_t numflag, int32_t ncon,
             nb::ndarray<float> tpwgts, nb::ndarray<float> ubvec, float itr,
             nb::ndarray<int32_t> part) {
            self.callParMETIS(vtxdist.data(), xadj.data(), adjncy.data(), vwgt.data(), vsize.data(),
                              adjwgt.data(), wgtflag, numflag, ncon, tpwgts.data(), ubvec.data(),
                              itr, part.data());
          },
          "vtxdist"_a, "xadj"_a, "adjncy"_a, "vwgt"_a, "vsize"_a, "adjwgt"_a, "wgtflag"_a,
          "numflag"_a, "ncon"_a, "tpwgts"_a, "ubvec"_a, "itr"_a, "part"_a);
}