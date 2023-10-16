################################################################################
# Specify tools
################################################################################

CP = cp
TAR = tar
ZIP = zip
MKDIR = mkdir
ZIPINFO = zipinfo

TARGET_BINS :=
TARGET_TESTS :=

CLEAN_BINS :=
CLEAN_DIRS :=

################################################################################
# Specify the PHONY targets
################################################################################

.PHONY: all
all: default

################################################################################
# Specify the target
################################################################################

TARGET_BIN := r11922001.zip
$(TARGET_BIN) : r11922001/DP_solver.py \
                r11922001/DP_solver_2_2.py \
                r11922001/Report.pdf
	@echo "Zipping $@ ..."
	$(Q)$(ZIP) -r $@ r11922001

TARGET_BINS += $(TARGET_BIN)
CLEAN_BINS += $(TARGET_BIN)
CLEAN_DIRS += r11922001

# -----------------------------------------------------------------------------

.PHONY: tests.tgz

TARGET_BIN := tests.tgz
$(TARGET_BIN): tests
	$(Q)$(TAR) -zcvf $@ $^

TARGET_TESTS += $(TARGET_BIN)
CLEAN_BINS += $(TARGET_BIN)

################################################################################
# Specify the dependencies
################################################################################

r11922001/DP_solver.py : prediction/DP_solver.py | r11922001
	@echo "Updating $^ ..."
	$(Q)$(CP) $^ $@

r11922001/DP_solver_2_2.py : control/DP_solver_2_2.py | r11922001
	@echo "Updating $^ ..."
	$(Q)$(CP) $^ $@

r11922001/Report.pdf : Report.pdf | r11922001
	$(Q)$(CP) $^ $@

r11922001:
	$(Q)$(MKDIR) -p $@

################################################################################
# Specify the PHONY targets
################################################################################

.PHONY: default
default: $(TARGET_BINS)
	@echo "Checking $^ ..."
	@echo ""
	$(Q)$(ZIPINFO) r11922001.zip

.PHONY: test
test: tests.tgz

.PHONY: clean
clean:
	$(Q)rm -rf $(CLEAN_BINS) $(CLEAN_DIRS)

################################################################################
# Debugging options
################################################################################

V ?= 0

ifeq ($(V), 0)
Q = @
else
Q =
endif