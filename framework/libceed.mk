ifneq ($(KOKKOS),true)
  ifeq ($(LIBCEED),true)
    $(error libCEED support requires Kokkos support to be enabled first)
  endif
endif

ifeq ($(LIBCEED),true)

ifeq ($(wildcard $(MOOSE_DIR)/libCEED/Makefile),)
  $(error libCEED submodule was not initialized)
endif

ifneq ($(CUDA_EXISTS),)
  LIBCEED_BACKEND := CUDA_DIR=$(shell dirname $(shell dirname $(CUDA_EXISTS)))
else ifneq ($(HIP_EXISTS),)
  LIBCEED_BACKEND := ROCM_DIR=$(shell dirname $(shell dirname $(HIP_EXISTS)))
else ifneq ($(SYCL_EXISTS),)
  LIBCEED_BACKEND := SYCL_DIR=$(shell dirname $(shell dirname $(SYCL_EXISTS)))
endif

libceed-build:
	make -C $(MOOSE_DIR)/libCEED CC=$(libmesh_CC) CXX=$(libmesh_CXX) $(LIBCEED_BACKEND)

libceed-clean:
	make -C $(MOOSE_DIR)/libCEED clean

LIBCEED_BUILD    := libceed-build
LIBCEED_CLEAN    := libceed-clean
LIBCEED_LDFLAGS  := -L$(MOOSE_DIR)/libCEED/lib -lceed

libmesh_INCLUDE  += -I$(MOOSE_DIR)/libCEED/include
libmesh_LDFLAGS  += -Wl,-rpath,$(MOOSE_DIR)/libCEED/lib

LIBCEED_JIT_INCLUDE := {
LIBCEED_JIT_INCLUDE += $(shell \
  paths="$(patsubst -I%,%,$(libmesh_INCLUDE))"; \
  last=$$(echo $$paths | awk '{print $$NF}'); \
  for path in $$paths; do \
    if [ "$$path" != "$$last" ]; then \
      echo \\\"$$path\\\"\\\\,; \
    else \
      echo \\\"$$path\\\"; \
    fi; \
  done; \
)
LIBCEED_JIT_INCLUDE += }

libmesh_CXXFLAGS += -DLIBCEED_JIT_INCLUDE=$(shell echo \'$(LIBCEED_JIT_INCLUDE)\')

else

libmesh_CXXFLAGS += -DMOOSE_IGNORE_LIBCEED

endif
