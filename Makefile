cpu_count=$(shell python -c 'import multiprocessing as m; print m.cpu_count()')

nose_env=NOSE_PROCESSES=$(cpu_count) NOSE_PROCESS_TIMEOUT=240

all: test

debug: FORCE
	mkdir -p build/debug
	cd build/debug \
	  && cmake -DCMAKE_BUILD_TYPE=Debug ../.. \
	  && $(MAKE)

release: FORCE
	mkdir -p build/release
	cd build/release \
	  && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../.. \
	  && $(MAKE)

install: debug release FORCE
	pip install -e .

clean: FORCE
	git clean -xdf -e loom.egg-info -e data

test: install
	@pyflakes loom/schema_pb2.py \
	  || (echo '...patching schema_pb2.py' \
	    ; sed -i '/descriptor_pb2/d' loom/schema_pb2.py)  # HACK
	pyflakes setup.py loom
	pep8 --repeat --ignore=E265 --exclude=*_pb2.py setup.py loom
	python -m loom.datasets init
	$(nose_env) nosetests -v loom
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

FORCE:
