#makefile for queue
all : queue queue_hand

queue : queue.cu
	nvcc -arch=sm_20 -gencode arch=compute_20,code=sm_20 queue.cu -o queue -g -G
#	nvcc -g -G -arch=sm_20 queue.cu -o queue
queue_hand : queue_hand
	nvcc -arch=sm_20 -gencode arch=compute_20,code=sm_20 queue_hand.cu -o queue_hand -g -G
clean : 
	rm queue queue_hand
