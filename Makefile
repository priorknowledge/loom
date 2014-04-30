cpu_count=$(shell python -c 'import multiprocessing as m; print m.cpu_count()')

nose_env=NOSE_PROCESSES=$(cpu_count) NOSE_PROCESS_TIMEOUT=120

all: test

debug: FORCE
	mkdir -p build/debug
	cd build/debug \
	  && CXX_FLAGS="$(CXX_FLAGS) -DDIST_DEBUG_LEVEL=3 -DLOOM_DEBUG_LEVEL=3" cmake -DCMAKE_BUILD_TYPE=Debug ../.. \
	  && $(MAKE)

release: FORCE
	mkdir -p build/release
	cd build/release \
	  && CXX_FLAGS="$(CXX_FLAGS) -DNDEBUG" cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../.. \
	  && $(MAKE)

build: debug release FORCE
	pip install -e .

clean: FORCE
	git clean -xdf -e loom.egg-info -e data

test: build
	@pyflakes loom/schema_pb2.py \
	  || (echo '...patching schema_pb2.py' \
	    ; sed -i '/descriptor_pb2/d' loom/schema_pb2.py)  # HACK
	pyflakes setup.py loom
	pep8 --repeat --ignore=E265 --exclude=*_pb2.py setup.py loom
	$(nose_env) nosetests -v loom
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

FORCE:
