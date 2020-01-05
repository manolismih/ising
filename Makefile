FLAGS = -O3
INCLUDES = -I ./inc

########################################################################

clean:
	find -name "*.o" -or -name "*.run" | xargs rm -f
	
test_v0: v0
	./v0.run
	diff test/conf-11.bin test/conf-11.ans
	@echo 'Test Passed!!!'
test_v1: v1
	./v1.run
	diff test/conf-11.bin test/conf-11.ans
	@echo 'Test Passed!!!'
test_v2: v2
	./v2.run
	diff test/conf-11.bin test/conf-11.ans
	@echo 'Test Passed!!!'
test_v3: v3
	./v3.run
	diff test/conf-11.bin test/conf-11.ans
	@echo 'Test Passed!!!'

v0: V0_sequential.o main.o
	nvcc $(FLAGS) $(INCLUDES) V0_sequential.o main.o -o v0.run
v1: V1_gpu1ThreadPerMoment.o main.o
	nvcc $(FLAGS) $(INCLUDES) V1_gpu1ThreadPerMoment.o main.o -o v1.run
v2: V2_gpu1ThreadPerBlockOfMoments.o main.o
	nvcc $(FLAGS) $(INCLUDES) V2_gpu1ThreadPerBlockOfMoments.o main.o -o v2.run
v3: V3_gpuMultipleThreads.o main.o
	nvcc $(FLAGS) $(INCLUDES) V3_gpuMultipleThreads.o main.o -o v3.run

%.o: src/%.cu
	nvcc $(FLAGS) $(INCLUDES) -c $<
