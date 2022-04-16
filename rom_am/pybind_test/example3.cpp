#include <iostream>
#include <stdlib.h>
#include <Python.h>

int main()
{
	PyObject* pInt;

	Py_Initialize();

	PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
	
	Py_Finalize();

	printf("\nPress any key to exit...\n");
	return 0;
}