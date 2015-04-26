prog_NAME := gaussnewton.py

.PHONY: all, clean

all:
	./$(prog_NAME)

clean:
	$(RM) *.pyc
