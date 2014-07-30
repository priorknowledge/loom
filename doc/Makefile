all: dataflow.png

dataflow.png: dataflow.dot FORCE
	python -m loom.documented make-dataflow
	dot dataflow.dot -T png -Gdpi=72 -o dataflow.png
	#dot dataflow.dot -T svgz -o dataflow.svgz

FORCE:
