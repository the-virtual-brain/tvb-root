edit_linux: linux_edit
linux_edit: 
	texmaker *.tex &
	gedit Makefile &

run:	
	pdflatex network.tex

# cleaning macros
touch:
	touch *.tex *py

clean_all: clean_garbage clean_lock

clean_garbage:
	rm -f *.fff *.lof *.dvi *.ps *.out *.log *.aux *.bbl *.blg *.snm *.nav *.toc *.synctex.gz *.fdb_latexmk

%.jpg: %.pdf
	convert $< $@

.PHONY: clean_all clean_garbage

