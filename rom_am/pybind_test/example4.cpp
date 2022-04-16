#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/LU>

namespace py = pybind11;

    Eigen::MatrixXd inv(const Eigen::MatrixXd &xs)
    {
    return xs.inverse();
    }

    double det(const Eigen::MatrixXd &xs)
    {
    return xs.determinant();
    }

    // ----------------
    // Python interface
    // ----------------


    PYBIND11_MODULE(ex4m,m)
    {
    m.doc() = "pybind11 example plugin";

    m.def("inv", &inv);

    m.def("det", &det);
    }

int main() {
    py::scoped_interpreter guard{};

    // Calculate e^π in decimal
    py::object Decimal = py::module_::import("decimal").attr("Decimal");
    py::object pi = Decimal("3.14159");
    py::print(pi);

    py::object dmd = py::module_::import("rom_am").attr("DMD")();

    py::print("done1");
    //py::module_ sys = py::module_::import("ex4");
    py::module_ np = py::module_::import("numpy");
    py::exec(R"(
        input_ = numpy.array([0, 1, 2])
    )");

    py::exec(R"(
        input_ = 2*input_
        print(input_)
    )");

    py::print("done2");


}
