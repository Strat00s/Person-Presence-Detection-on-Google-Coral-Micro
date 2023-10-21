git clone --recurse-submodules -j8 https://github.com/google-coral/coralmicro


## Taken from https://coral.ai/docs/dev-board-micro/freertos/#create-an-in-tree-project
cd coralmicro

# Copy the "hello world" code
cp -r examples/hello_world apps/my_project

# Rename the executable
mv apps/my_project/hello_world.cc apps/my_project/main.cc


#add_executable_m7(my_project
#    main.cc
#)
#
#target_link_libraries(my_project
#    libs_base-m7_freertos
#)

#add_subdirectory(my_project)

#change hidapi version
#hidapi==0.11.2

#bash setup.sh