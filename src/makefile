CC      = gcc
CFLAGS  = -Wall -O3 -std=c11 
LDFLAGS = -lm -lpthread

BETH_SRCS    = ../../lib
APP_SRCS     = ./

BETH_BCORE_SRCS = $(BETH_SRCS)/bcore
BETH_BCLOS_SRCS = $(BETH_SRCS)/bclos
BETH_BMATH_SRCS = $(BETH_SRCS)/bmath

INCLUDES = \
    -I $(BETH_BCORE_SRCS) \
    -I $(BETH_BCLOS_SRCS) \
    -I $(BETH_BMATH_SRCS)

SRCS = \
    $(wildcard $(BETH_BCORE_SRCS)/*.c) \
    $(wildcard $(BETH_BCLOS_SRCS)/*.c) \
    $(wildcard $(BETH_BMATH_SRCS)/*.c) \
    $(wildcard $(APP_SRCS)/*.c)
    
HDRS = \
    $(wildcard $(BETH_BCORE_SRCS)/*.h) \
    $(wildcard $(BETH_BCLOS_SRCS)/*.h) \
    $(wildcard $(BETH_BMATH_SRCS)/*.h) \
    $(wildcard $(APP_SRCS)/*.h)

TARGET = bmath_dev

$(TARGET): $(SRCS) $(HDRS)
ifeq ($(wildcard $(BETH_BCORE_SRCS)/*.c),)
	@echo "Sources of beth-library missing: $(BETH_BCORE_SRCS). Clone from https://github.com/johsteffens/beth.git"
else
	$(CC) -o $@ $(SRCS) $(CFLAGS) $(INCLUDES) $(LDFLAGS) 
endif

.PHONY: clean

clean:
	rm -f $(TARGET)
