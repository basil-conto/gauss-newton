prog_GAUSS := gaussnewton.py
prog_GRAPH := graph.py

.PHONY: all, gauss, clean, graph

all: gauss graph

gauss: ; ./$(prog_GAUSS)

graph: ; ./$(prog_GRAPH)

clean: ; $(RM) *.pyc
