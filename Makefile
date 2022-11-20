TARGET = simpleAnn

CC := gcc
CFLAGS := -g -Wall

SRCS := $(wildcard *.c)
OBJS := $(SRCS:%.c=%.o)
INC_DIRS := -I ./
	
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ -lm
	
%.o: %.cpp $(INC_DIRS)
	$(CC) $(CFLAGS) $@ -c $^ -lm

clean:
	rm -rf *.o *.gch simpleAnn
