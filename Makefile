CC = mpic++
CFLAGS = -std=c++17 -O3 -fopenmp
LDFLAGS = -lmetis
INCLUDE = -Iinclude/
SRC = src/main.cpp src/sssp_update.cpp src/utils.cpp
TARGET = sssp_update

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDE) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

