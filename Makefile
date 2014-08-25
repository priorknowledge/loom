cpu_count=$(shell python -c 'import multiprocessing as m; print m.cpu_count()')

nose_env=
ifndef NOSE_PROCESSES
	nose_env+=NOSE_PROCESSES=$(cpu_count) NOSE_PROCESS_TIMEOUT=1800
endif

cmake_args=
cmake_env=
ifdef VIRTUAL_ENV
	cmake_args+=-DCMAKE_INSTALL_PREFIX=$(VIRTUAL_ENV)
	cmake_env+=CMAKE_PREFIX_PATH=$(VIRTUAL_ENV)
endif
cmake = $(cmake_env) cmake $(cmake_args)

all: debug release

debug: FORCE
	mkdir -p build/debug
	cd build/debug \
	  && $(cmake) -DCMAKE_BUILD_TYPE=Debug ../.. \
	  && $(MAKE)

release: FORCE
	mkdir -p build/release
	cd build/release \
	  && $(cmake) -DCMAKE_BUILD_TYPE=Release ../.. \
	  && $(MAKE)

install_cc: debug release FORCE
	cd build/release && $(MAKE) install
	cd build/debug && $(MAKE) install

dev: install_cc FORCE
	pip install -e .

install: install_cc FORCE
	pip install .

package: release FORCE
	cd build/release && $(MAKE) package
	mv build/release/loom.tar.gz build/

clean: FORCE
	git clean -xdf -e loom.egg-info -e data

data: dev FORCE
	python -m loom.datasets init

test: dev
	@pyflakes loom/schema_pb2.py \
	  || (echo '...patching schema_pb2.py' \
	    ; sed -i '/descriptor_pb2/d' loom/schema_pb2.py)  # HACK
	pyflakes setup.py loom examples
	pep8 --repeat --exclude=*_pb2.py setup.py loom examples
	python -m loom.datasets test
	$(nose_env) nosetests -v loom examples/taxi
	$(MAKE) -C doc
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

small-test:
	LOOM_TEST_COST=100 $(MAKE) test

big-test:
	LOOM_TEST_COST=1000 $(MAKE) test

bigger-test:
	LOOM_TEST_COST=10000 $(MAKE) test

license: FORCE
	python update_license.py update

FORCE:
