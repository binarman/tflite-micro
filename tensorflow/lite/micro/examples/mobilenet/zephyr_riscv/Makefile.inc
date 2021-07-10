ifeq ($(TARGET), zephyr_vexriscv)
	export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
	export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
	export TOOLCHAIN_VERSION=9.2.0
	PROJECT_INCLUDES += ${CURDIR} ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} ${TOOLCHAIN_BASE}/include ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
	ZEPHYR_MOBILENET_SRCS = \
tensorflow/lite/micro/examples/mobilenet/zephyr_riscv/src/assert.cc \
tensorflow/lite/micro/examples/mobilenet/main.cc \
tensorflow/lite/micro/examples/mobilenet/main_functions.cc \
tensorflow/lite/micro/examples/mobilenet/constants.cc \
tensorflow/lite/micro/examples/mobilenet/output_handler.cc \
tensorflow/lite/micro/examples/mobilenet/model.cc \
prj.conf

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,mobilenet,$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_MOBILENET_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MOBILENET_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)mobilenet/cmake/CMakeLists.txt: $(PRJDIR)mobilenet/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

#We are skipping here copy of `zephyr` third_party repository
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
mobilenet_bin: generate_mobilenet_cmake_project $(PRJDIR)mobilenet/cmake/CMakeLists.txt
	( \
	  . ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
	  cmake -B${GENDIR}mobilenet/build -DBOARD="litex_vexriscv" -H${PRJDIR}mobilenet/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
	  make -C ${GENDIR}mobilenet/build; \
	)
endif
