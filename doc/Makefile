DUMMY := $(shell latexmk --version 2>/dev/null)

all: gen1

gen1: content.md 
	pandoc content.md -r markdown-auto_identifiers -o content.tex --listings
	pdflatex --interaction=batchmode doc.tex 2>&1 > /dev/null
	pdflatex --interaction=batchmode doc.tex 2>&1 > /dev/null
ifdef DUMMY
	latexmk -c -quiet
else
endif

