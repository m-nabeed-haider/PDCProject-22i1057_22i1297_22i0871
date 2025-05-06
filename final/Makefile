CC = mpic++
CFLAGS = -std=c++17 -O3 -fopenmp
LDFLAGS = -lmetis
INCLUDE = -Iinclude/
SRC = main.cpp sssp_update.cpp utils.cpp
TARGET = sssp_update

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDE) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

