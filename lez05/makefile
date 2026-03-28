CC = nvcc
CFLAGS = -arch=sm_89 
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: matmat.exe matmat_naive.exe

matmat.exe: matmat.obj utils.obj
	$(CC) matmat.obj utils.obj -o matmat.exe $(CFLAGS)

matmat_naive.exe: matmat_naive.obj utils.obj
	$(CC) matmat_naive.obj utils.obj -o matmat_naive.exe $(CFLAGS)

matmat.obj: matmat.cu utils.h
	$(CC) -c matmat.cu -o matmat.obj $(CFLAGS)

matmat_naive.obj: matmat_naive.cu utils.h
	$(CC) -c matmat_naive.cu -o matmat_naive.obj $(CFLAGS)

utils.obj: utils.c utils.h
	$(CC) -c utils.c -o utils.obj $(CFLAGS)

clean:
	rm -f *.exe *.obj *.pdb