all: dataflow.png

dataflow.png: dataflow.dot FORCE
	python -m loom.documented make-dataflow
	dot dataflow.dot -T svg -o dataflow.svg

FORCE:
