CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude -Ithird_party
LDLIBS ?= -lm

SOURCES = src/image_io.c src/tensor_io.c src/layers.c src/resnet18.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean test

all: resnet18

resnet18: $(OBJECTS) src/main.o
	$(CC) -o $@ $^ $(LDLIBS)

test_forward: $(OBJECTS) tests/test_forward.o
	$(CC) -o $@ $^ $(LDLIBS)

test: test_forward
	./test_forward weights/resnet18.bin

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) $(OBJECTS) src/main.o tests/test_forward.o resnet18 test_forward resnet18.exe test_forward.exe
