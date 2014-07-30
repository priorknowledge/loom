all: dataflow.png

dataflow.png: dataflow.dot FORCE
	python -m loom.documented make-dataflow
	dot dataflow.dot -T svg -o dataflow.svg
	dot dataflow.dot -T svgz -o dataflow.svgz
	dot dataflow.dot -T png -o dataflow.png
	dot dataflow.dot -T eps -o dataflow.eps
	dot dataflow.dot -T pdf -o dataflow.pdf

FORCE:
