#makefile for queue
queue : queue.cu
	nvcc -arch=sm_20 -gencode arch=compute_20,code=sm_20 queue.cu -o queue -g -G
#	nvcc -g -G -arch=sm_20 queue.cu -o queue
clean : 
	rm queue
