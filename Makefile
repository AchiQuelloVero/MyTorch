NAME	=	my_torch_analyzer

SRC	=	src/analyzer.py

GENERATOR	=	my_torch_generator

GENERATOR_SRC	=	src/generator.py

all:	$(NAME) $(GENERATOR)

$(NAME):	$(SRC)
	cp $(SRC) $(NAME)
	chmod +x $(NAME)

$(GENERATOR):	$(GENERATOR_SRC)
	cp $(GENERATOR_SRC) $(GENERATOR)
	chmod +x $(GENERATOR)

clean:
	rm -rf src/__pycache__ src/*/__pycache__ tests/__pycache__

fclean:	clean
	rm -f $(NAME) $(GENERATOR)

re:	fclean all

tests_run:
	python3 -m unittest discover -s tests -v
