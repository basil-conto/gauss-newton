prog_GAUSS := gaussnewton.py
prog_GRAPH := graph.py

.PHONY: all, gauss, clean, graph

all: gauss graph

gauss: ; pipenv run python $(prog_GAUSS)

graph: ; pipenv run python $(prog_GRAPH)
