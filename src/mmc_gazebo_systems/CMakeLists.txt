cmake_minimum_required(VERSION 3.16)

project(mmc_gazebo_systems)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)

add_library(mmc-wind-velocity-system SHARED
  src/wind_velocity_publisher.cc
)

target_compile_features(mmc-wind-velocity-system PUBLIC cxx_std_17)
target_link_libraries(mmc-wind-velocity-system
  gz-sim8::gz-sim8
  gz-msgs10::gz-msgs10
  gz-transport13::gz-transport13
)

install(
  TARGETS mmc-wind-velocity-system
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

add_library(mmc-slider-servo-system SHARED src/slider_servo.cc)
target_link_libraries(mmc-slider-servo-system
  gz-sim8::gz-sim8 gz-msgs10::gz-msgs10 gz-transport13::gz-transport13)
install(TARGETS mmc-slider-servo-system LIBRARY DESTINATION lib)

# Make the plugin discoverable after sourcing the workspace setup file.  The
# shared object is installed directly under the package prefix's lib directory.
ament_environment_hooks(
  environment/gz_sim_system_plugin_path.sh.in
  environment/gz_sim_system_plugin_path.dsv.in
)

ament_export_dependencies(gz-msgs10 gz-sim8 gz-transport13)
ament_package()
