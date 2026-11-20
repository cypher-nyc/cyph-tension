# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# The productive-tension model in Haskell (see ../README.md).
#
#   make lib     build/libcyphtension.{dylib,so}: the C-callable library the
#                Python package loads (tension/_haskell.py)
#   make test    the property tests (test/Properties.hs, test/PairProperties.hs)
#   make figure13  rewrite test/golden/figure13.tsv, the figure's curve from
#                Cyph.Pair (the golden test then holds the model to it)
#   make bundle  (Linux) build/dist/: the library and the GHC runtime
#                libraries it needs, side by side, for the image
#   make clean
#
# Plain GHC, no cabal and no packages: base only. GHC from ghcup lives in
# ~/.ghcup/bin when it is not on PATH.

GHC ?= $(shell command -v ghc 2>/dev/null || echo $(HOME)/.ghcup/bin/ghc)
UNAME := $(shell uname -s)
LIB_EXT := $(if $(filter Darwin,$(UNAME)),dylib,so)
LIB := build/libcyphtension.$(LIB_EXT)
SRC := src/Cyph/Tension.hs src/Cyph/TensionFFI.hs src/Cyph/Pair.hs
# The library links GHC's runtime libraries dynamically. On Linux it also
# looks for them beside itself ($ORIGIN), where `make bundle` puts them.
RPATH := $(if $(filter Darwin,$(UNAME)),,-optl-Wl,-rpath,'$$ORIGIN')

.PHONY: lib test figure13 bundle clean

lib: $(LIB)

# -shared + -flink-rts: a shared library carrying the Haskell runtime's
# entry points; the caller starts the runtime once with hs_init.
$(LIB): $(SRC)
	@mkdir -p build/obj
	$(GHC) -O2 -Wall -fPIC -shared -dynamic -flink-rts $(RPATH) -isrc \
	  -odir build/obj -hidir build/obj -o $@ $(SRC)

test: build/properties build/pair-properties
	./build/properties
	./build/pair-properties

figure13: build/pair-properties
	@mkdir -p test/golden
	./build/pair-properties --figure13 > test/golden/figure13.tsv

build/properties: src/Cyph/Tension.hs test/Properties.hs
	@mkdir -p build/test
	$(GHC) -O2 -isrc -odir build/test -hidir build/test -o $@ test/Properties.hs

build/pair-properties: src/Cyph/Tension.hs src/Cyph/Pair.hs test/PairProperties.hs
	@mkdir -p build/pair-test
	$(GHC) -O2 -isrc -odir build/pair-test -hidir build/pair-test -o $@ test/PairProperties.hs

# The library plus every libHS*.so it loads, so a runtime image needs no GHC
# install (only libgmp, which the image installs).
bundle: $(LIB)
	rm -rf build/dist && mkdir -p build/dist
	cp $(LIB) build/dist/
	ldd $(LIB) | awk '/libHS/ {print $$3}' | xargs -r -I{} cp {} build/dist/
	ls -la build/dist

clean:
	rm -rf build
