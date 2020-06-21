// Developer: Wilbert (wilbert.phen@gmail.com)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "model.h"
#include "embeddings.h"
#include "datasets.h"
#include "tagger.h"

namespace py = pybind11;

PYBIND11_MODULE(_tagger, m) {
    py::class_<Tagger>(m, "Tagger")
        .def(py::init<>())
        .def(py::init<int, int, int, float, std::string, std::string, std::string, std::string>())
        .def("train", &Tagger::train)
        .def("test", &Tagger::test);
}
