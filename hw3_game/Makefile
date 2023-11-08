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
$(TARGET_BIN) : r11922001/gridworld.py \
                r11922001/train.py \
				r11922001/my2048_env.py \
				r11922001/model.zip \
                r11922001/report.pdf
	@echo "Zipping $@ ..."
	$(Q)$(ZIP) -r $@ r11922001

TARGET_BINS += $(TARGET_BIN)
CLEAN_BINS += $(TARGET_BIN)
CLEAN_DIRS +=

################################################################################
# Specify the dependencies
################################################################################

r11922001/gridworld.py : grid_world/gridworld.py | r11922001
	@echo "Updating $^ ..."
	$(Q)$(CP) $^ $@

r11922001/train.py : 2048/train.py | r11922001
	@echo "Updating $^ ..."
	$(Q)$(CP) $^ $@

r11922001/my2048_env.py : 2048/envs/my2048_env.py | r11922001
	@echo "Updating $^ ..."
	$(Q)$(CP) $^ $@

r11922001:
	$(Q)$(MKDIR) -p $@

CLEAN_BINS += $(wildcard r11922001/*.py)

################################################################################
# Specify the PHONY targets
################################################################################

.PHONY: default
default: $(TARGET_BINS)
	@echo "Checking $^ ..."
	@echo ""
	$(Q)$(ZIPINFO) r11922001.zip

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