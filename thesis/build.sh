# This script compiles the LaTeX document for the thesis using latexmk, which automates the process of running LaTeX the necessary number of times to resolve references and citations.
# Usage: Run this script from the terminal in the thesis directory to generate the PDF output in the 'output' folder.
#!/bin/bash
latexmk -pdf -output-directory=output main.tex