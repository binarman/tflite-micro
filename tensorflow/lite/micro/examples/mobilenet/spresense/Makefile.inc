# Settings for Spresense platform for Hello World example
# This should be read when the EXTERNALS_TENSORFLOW_EXAMPLE_HELLOWORLD option is selected
# in Spresense configuration.

ifeq ($(TARGET), spresense)
ifeq ($(CONFIG_EXTERNALS_TENSORFLOW_EXAMPLE_HELLOWORLD),y)

SPRESENSE_MOBILENET_EXCLUDED_SRCS = \
    tensorflow/lite/micro/examples/mobilenet/main.cc

SPRESENSE_MOBILENET_SRCS = \
    $(filter-out $(SPRESENSE_MOBILENET_EXCLUDED_SRCS),$(MOBILENET_SRCS))

# In spresence case, those file should be included into libtensorflow-microlite.
THIRD_PARTY_CC_SRCS += $(SPRESENSE_MOBILENET_SRCS)

endif
endif
