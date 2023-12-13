LATEX	= pdflatex
BIBTEX	= bibtex

TARGET	= main

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).bbl: *.bib
	$(LATEX)	$(TARGET)
	$(BIBTEX)	$(TARGET)

$(TARGET).pdf: *.tex $(TARGET).bbl
	$(LATEX)	$(TARGET)
	$(LATEX)	$(TARGET)

clean:
	$(RM) $(addprefix $(TARGET)., aux bbl blg ent log out pdf synctex.gz)
