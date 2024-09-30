TARGET = testbench
OBJS   = testbench.o testData.o gemm_cpu_fp.o
CXXFLAGS += -Wall -g 
LDLIBS = -lm

# just define the dependencies, let makefile defaults do the rest
$(TARGET): $(OBJS)

# for test just execute the testbench
test: $(TARGET)
	./$(TARGET)

# remove objs and target 
clean: 
	rm -f $(TARGET) $(OBJS)
