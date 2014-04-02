all:
	mkdir -p build
	cd build && cmake .. && $(MAKE)

clean: FORCE
	git clean -Xdf

FORCE:
