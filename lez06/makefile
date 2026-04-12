CC = nvcc
CFLAGS = -arch=sm_89
XFLAGS = -Xcompiler=/openmp
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: main.exe

main.exe: integrals.obj trapezoid.obj main.obj
	$(CC) integrals.obj trapezoid.obj main.obj -o main.exe $(CFLAGS) $(XFLAGS)

main.obj: main.cu integrals.hpp
	$(CC) -c main.cu -o main.obj $(CFLAGS) $(XFLAGS)

integrals.obj: integrals.cu integrals.hpp trapezoid.hpp utils.hpp
	$(CC) -c integrals.cu -o integrals.obj $(CFLAGS) $(XFLAGS)

trapezoid.obj: trapezoid.cu trapezoid.hpp utils.hpp
	$(CC) -c trapezoid.cu -o trapezoid.obj $(CFLAGS) $(XFLAGS)

clean:
	rm -f *.exe *.obj *.pdb