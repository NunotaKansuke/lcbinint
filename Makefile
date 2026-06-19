BUILD_DIR ?= build
CMAKE     ?= cmake
CTEST     ?= ctest

.PHONY: all configure core python test clean

all: core

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

core: configure
	$(CMAKE) --build $(BUILD_DIR) --target lcbinint_core

python: configure
	$(CMAKE) --build $(BUILD_DIR) --target lcbinint_python

test: configure
	$(CMAKE) --build $(BUILD_DIR) --target test_core
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)
