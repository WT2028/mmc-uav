cmake_minimum_required(VERSION 3.8)
project(mmc_uav_description)

find_package(ament_cmake REQUIRED)

foreach(dir config meshes urdf worlds)
  install(DIRECTORY ${dir}/
    DESTINATION share/${PROJECT_NAME}/${dir}
  )
endforeach()

ament_package()
