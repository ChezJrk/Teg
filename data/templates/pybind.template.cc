#include <pybind11/pybind11.h>
#include "%C_FILENAME%"

PYBIND11_MODULE(%MODULE_NAME%, m) {
    m.doc() = "";
    m.def("%FN_PYTHON_NAME%", &%FN_C_NAME%, "");
}