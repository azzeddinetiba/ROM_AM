#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

int main() {
    // Calculate e^π in decimal
    py::object Decimal = py::module_::import("decimal").attr("Decimal");
    py::object pi = Decimal("3.14159");
    py::object exp_pi = pi.attr("exp")();
    py::print(py::str(exp_pi));
}
