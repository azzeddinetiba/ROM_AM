#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/LU>

namespace py = pybind11;
using namespace py::literals;

    Eigen::MatrixXd inv(const Eigen::MatrixXd &xs)
    {
    return xs.inverse();
    }

    double det(const Eigen::MatrixXd &xs)
    {
    return xs.determinant();
    }

    double give()
    {
    double g = 90.2;
    return g;
    }

    // ----------------
    // Python interface
    // ----------------


    PYBIND11_MODULE(ex4m,m)
    {
    m.doc() = "pybind11 ex4m plugin";

    m.def("inv", &inv);

    m.def("det", &det);

    m.def("g", &give);
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
    py::exec(R"(
        import numpy
        input_ = numpy.array([0, 1, 2])
    )");

    py::exec(R"(
        input_ = 2*input_
        print(input_)
    )");

    py::print("done2");

    auto locals = py::dict();
    py::exec(R"(
        import ex4m
        import numpy
        res = ex4m.g()
        print(res)
        res2 = numpy.array([1, 2, 3])
    )", py::globals(), locals);

    std::cout<<locals["res"].cast<double>();
    std::cout<<locals["res2"].cast<Eigen::MatrixXd>();

    py::object result = locals["res2"];
    py::module_ np = py::module_::import("numpy");
    py::object result2 = np.attr("exp")(result);

    std::cout<<"\n tst1 \n";
    Eigen::MatrixXf interm(2, 2);
	interm(0, 0) = -1;
	interm(0, 1) = -1;
	interm(1, 0) = 1;
	interm(1, 1) = 1;
    py::object result3 = np.attr("exp")(interm);
    std::cout<<"\n tst2 \n";

    std::cout<<result2.cast<Eigen::MatrixXd>();

    py::exec(R"(
        import rom_am
        import numpy as np
        ll = rom_am.DMD()
        bb = rom_am.ROM(ll)
        bb.decompose(X = np.random.randn(12, 4), Y = np.random.randn(12, 4), dt = 0.4)
    )", py::globals(), locals);

    py::object res_dmd = locals["ll"];
    py::object res_dmd_2 = res_dmd.attr("singvals");

    std::cout<<"YOO \n";
    std::cout<<res_dmd_2.cast<Eigen::VectorXd>();

}
