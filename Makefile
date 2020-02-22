CC=nvcc
SOURCES=cuda.cu
#OBJECTS=$(SOURCES:.cu=.o)
CUFLAG = 


cuda: cuda.cu
	nvcc -o cuda cuda.cu

test:
	./cuda

# for GDB -o data/predictions.tsv -i data/ML1M/u1.base -y data/ML1M/u1.test -t 16
# for GDB -o data/predictions.tsv -i data/Netflix/netflix_cumf_train.tsv -y data/Netflix/netflix_cumf_test.tsv


