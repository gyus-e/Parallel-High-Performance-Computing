CC = nvcc
CFLAGS = -arch=sm_89 
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: matmat.exe matmat_naive.exe matmatthread.exe

matmat.exe: matmat.obj utils.obj
	$(CC) matmat.obj utils.obj -o matmat.exe $(CFLAGS)

matmat_naive.exe: matmat_naive.obj utils.obj
	$(CC) matmat_naive.obj utils.obj -o matmat_naive.exe $(CFLAGS)

matmatthread.exe: matmatthread.obj utils.obj
	$(CC) matmatthread.obj utils.obj -o matmatthread.exe

matmat.obj: matmat.cu utils.hpp
	$(CC) -c matmat.cu -o matmat.obj $(CFLAGS)

matmat_naive.obj: matmat_naive.cu utils.hpp
	$(CC) -c matmat_naive.cu -o matmat_naive.obj $(CFLAGS)

matmatthread.obj: matmatthread.cpp utils.hpp
	$(CC) -c matmatthread.cpp -o matmatthread.obj

utils.obj: utils.cpp utils.hpp
	$(CC) -c utils.cpp -o utils.obj $(CFLAGS)

clean:
	rm -f *.exe *.obj *.pdb