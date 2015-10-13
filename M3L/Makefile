CC=g++
INCL=include
SRC=src
OBJ=obj
BIN=bin/linux
CXXFLAGS= -I$(INCL) -O3
LAINCL= $(INCL)/myvectors.h $(INCL)/definitions.h
TOOLSINCL= $(INCL)/cache.h $(INCL)/definitions.h
COMPLETEINCL= $(LAINCL) $(TOOLSINCL) $(INCL)/structures.h $(INCL)/M3LKernel.h $(INCL)/M3LLinear.h

all: $(BIN)/M3L
$(BIN)/M3L: $(OBJ)/structures.o $(OBJ)/M3LKernel.o $(OBJ)/M3LLinear.o $(OBJ)/myvectors.o $(OBJ)/cache.o $(OBJ)/main.o
	$(CC) -O3 -o $@ $^
$(OBJ)/main.o: $(SRC)/main.cpp $(COMPLETEINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/structures.o: $(SRC)/structures.cpp $(COMPLETEINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/M3LKernel.o: $(SRC)/M3LKernel.cpp $(COMPLETEINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/M3LLinear.o: $(SRC)/M3LLinear.cpp $(COMPLETEINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/matrix.o: $(SRC)/matrix.cpp $(LAINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/vectors.o: $(SRC)/vectors.cpp $(LAINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/cache.o: $(SRC)/cache.cpp $(TOOLSINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
$(OBJ)/myvectors.o: $(SRC)/myvectors.cpp $(LAINCL)
	$(CC) -c -o $@ $(CXXFLAGS) $<
clean:
	rm -rf $(OBJ)/*.o $(BIN)/M3L */*~
	
