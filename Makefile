cpu_count=$(shell python -c 'import multiprocessing as m; print m.cpu_count()')

nose_env=NOSE_PROCESSES=$(cpu_count) NOSE_PROCESS_TIMEOUT=1800

cmake_args=
cmake_env=
ifdef VIRTUAL_ENV
	cmake_args+=-DCMAKE_INSTALL_PREFIX=$(VIRTUAL_ENV)
	cmake_env+=CMAKE_PATH_PREFIX=$(VIRTUAL_ENV)
endif
cmake = $(cmake_env) cmake $(cmake_args)

all: test

debug: FORCE
	mkdir -p build/debug
	cd build/debug && $(cmake) -DCMAKE_BUILD_TYPE=Debug ../..  && $(MAKE)

release: FORCE
	mkdir -p build/release
	cd build/release && $(cmake) -DCMAKE_BUILD_TYPE=Release ../..  && $(MAKE)

install: debug release FORCE
	pip install -e .

package: release FORCE
	cd build/release && $(MAKE) package
	mv build/release/loom.tar.gz build/

clean: FORCE
	git clean -xdf -e loom.egg-info -e data

data: install FORCE
	python -m loom.datasets init

test: install data
	@pyflakes loom/schema_pb2.py \
	  || (echo '...patching schema_pb2.py' \
	    ; sed -i '/descriptor_pb2/d' loom/schema_pb2.py)  # HACK
	pyflakes setup.py loom
	pep8 --repeat --ignore=E265 --exclude=*_pb2.py setup.py loom
	$(nose_env) nosetests -v loom
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

FORCE:
