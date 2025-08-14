cmake_minimum_required(VERSION 3.0.2)
project(skrl_controller)

find_package(catkin_simple REQUIRED)

# catkin_python_setup()

catkin_simple()

file(GLOB python_scripts src/*.py)
catkin_install_python(PROGRAMS
    ${python_scripts} DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION})

cs_install()
cs_export()