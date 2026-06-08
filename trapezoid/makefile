SRC = ./src
BIN = ./bin
CC = nvcc
CFLAGS = -arch=sm_89
XFLAGS = -Xcompiler=-fopenmp
# Adjust the architecture flag as needed, e.g. -arch=sm_75 on Google Colaboratory 

all: main.exe

main.exe: $(BIN)/main.obj $(BIN)/integrals.obj $(BIN)/trapezoid.obj $(SRC)/utils.hpp
	$(CC) $(BIN)/main.obj $(BIN)/integrals.obj $(BIN)/trapezoid.obj -o main.exe $(CFLAGS) $(XFLAGS)

$(BIN)/main.obj: $(SRC)/main.cu $(SRC)/integrals.hpp $(SRC)/trapezoid.hpp $(SRC)/utils.hpp
	$(CC) -c $(SRC)/main.cu -o $(BIN)/main.obj $(CFLAGS) $(XFLAGS)

$(BIN)/integrals.obj: $(SRC)/integrals.cu $(SRC)/integrals.hpp $(SRC)/trapezoid.hpp $(SRC)/utils.hpp
	$(CC) -c $(SRC)/integrals.cu -o $(BIN)/integrals.obj $(CFLAGS) $(XFLAGS)

$(BIN)/trapezoid.obj: $(SRC)/trapezoid.cu $(SRC)/trapezoid.hpp $(SRC)/utils.hpp
	$(CC) -c $(SRC)/trapezoid.cu -o $(BIN)/trapezoid.obj $(CFLAGS) $(XFLAGS)

clean:
	rm -f *.exe *.obj *.pdb $(BIN)/*