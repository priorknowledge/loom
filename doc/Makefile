all: dataflow.png

dataflow.png: dataflow.dot FORCE
	python -m loom.documented make-dataflow
	dot dataflow.dot -Gdpi=72 -T png -o dataflow.png
	#dot dataflow.dot -Gdpi=72 -T ps -o dataflow.ps
	#gs -q -dNOPAUSE -dBATCH -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sDEVICE=png16m -sOutputFile=dataflow.png dataflow.ps

FORCE:
