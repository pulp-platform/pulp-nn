
# This makefile has been generated for a specific configuration and is bringing
# all flags and rules needed for compiling for this configuration.
# In case the rules are not suitable but flags are still needed, INCLUDE_NO_RULES
# can be defined in the application makefile.
# The 2 included makefiles can also be copied to application, customized and 
# included instead of the one from the SDK.

ifndef PULP_SDK_INSTALL
export PULP_SDK_INSTALL=$(TARGET_INSTALL_DIR)
export PULP_SDK_WS_INSTALL=$(INSTALL_DIR)
endif

-include /home/ilnaza/release_version/pulp_nn/test/build/pulp/__flags.mk

ifndef INCLUDE_NO_RULES
-include /home/ilnaza/release_version/pulp_nn/test/build/pulp/__rules.mk
endif
