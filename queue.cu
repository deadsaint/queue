#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct node{
	//TODO template for any data type
	int data;
	struct node * next;
}node, * pnode;

typedef struct queue{
	pnode head;
	pnode tail;
}queue, *pqueue;


__device__ int enqueue(int mydata,pqueue myqueue);
__device__ int dequeue(pnode mynode, pqueue myqueue);
__device__ pnode myAtomicCAS(pnode * address, pnode compare, pnode val);
__device__ void deleteNode(pnode delnode);

__global__ void app_bfs(pqueue myqueue);
__global__ void init(pqueue myqueue);
__global__ void show(pqueue myqueue);


int isError(cudaError_t cudaStatus, char* error_info);

int main(int argc, char * argv[]){
	int num_block, thread_per_block;
	pqueue d_myqueue;

	cudaError_t cudaStatus;
	
	if(argc != 3){// with this if we cant go into cuda debug
		printf("Usage: queue block_num thread_num\n");
		return -1;
	}
	num_block = atoi(argv[1]);
	thread_per_block = atoi(argv[2]);

	cudaStatus = cudaDeviceReset();
	if (isError(cudaStatus, "cudaDeviceReset error."))
		return -1;

	cudaStatus = cudaMalloc((void **)&d_myqueue, sizeof(queue));

	cudaEvent_t start, stop;
 	float elapsedTime;
 	cudaStatus = cudaEventCreate(&start);
 	cudaStatus = cudaEventCreate(&stop);
 	cudaStatus = cudaEventRecord(start, 0);

	init<<<1,1>>>(d_myqueue);
	cudaStatus = cudaDeviceSynchronize();
	app_bfs<<<num_block,thread_per_block>>>(d_myqueue);
	cudaStatus = cudaDeviceSynchronize();
	printf("[Info]%s\n",cudaGetErrorString(cudaGetLastError()));
	//show<<<1,1>>>(d_myqueue);
	//cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaEventRecord(stop, 0);
	cudaStatus = cudaEventSynchronize(stop);
	cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("[Info]Block:%d\tThread:%d\tElapsedTime:%fms\n", num_block, thread_per_block, elapsedTime); 

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_myqueue);
	//cudaDeviceSynchronize();
	//printf("[Info]%s\n",cudaGetErrorString(cudaGetLastError()));

	printf("[Info]Complete!\n");
	return 0;
}

__global__ void init(pqueue myqueue){
	pnode d_dummy = (pnode)malloc(sizeof(node));
	d_dummy->data = -1;
	d_dummy->next = NULL;
	myqueue->head = d_dummy;
	myqueue->tail = d_dummy;
}

int isError(cudaError_t cudaStatus, char* error_info)
{
	if (cudaStatus != cudaSuccess){
		printf("[Error]%s\n", error_info);
		return 1;
	}
	else
		return 0;
}



__global__ void show(pqueue myqueue){
   pnode temp = myqueue->head;
   while(temp != NULL){
   	printf("%d\t",temp->data);
   	temp = temp->next;
   }
   printf("\n");
	
}


__global__ void app_bfs(pqueue myqueue){

	pnode newnode = (pnode)malloc(sizeof(node));
	if(threadIdx.x % 2 == 1){
	//if(1){
		//printf("block:%d\tthread:%d\n", blockIdx.x, threadIdx.x);
		enqueue(blockIdx.x * blockDim.x + threadIdx.x, myqueue);
	}
	else{
		dequeue(newnode, myqueue);
	}
}

__device__ pnode myAtomicCAS(pnode * address, pnode compare, pnode val){
	//compare just the address, not the value.
	//sizeof(data *) = 8 int x64 and 4 in win32
	return (pnode)atomicCAS((unsigned long long int*)address, (unsigned long long int)compare, (unsigned long long int)val);
}


__device__ int enqueue(int newdata,pqueue myqueue){
	pnode tail = NULL,next = NULL;
	pnode newnode = (pnode)malloc(sizeof(node));
	int flag = sizeof(node);
	flag = sizeof(pnode);
	/*
	if (newnode == NULL){// added can avoid the unspecified launch failure!!!
		printf("[Error]Malloc failed!\n");
		return ;
	}
	*/
	newnode->data = newdata;
	newnode->next = NULL;
	
	while(1){
		tail = myqueue->tail;
		next = tail->next;
		if(tail == myqueue->tail){
			if(next == NULL){
				if(next == myAtomicCAS(&myqueue->tail->next, next, newnode)){
					flag = 1;
					break;
				}
			}
			else{ 
				myAtomicCAS(&myqueue->tail, tail, next);// success or not both ok
			}
		}
	}
	myAtomicCAS(&myqueue->tail, tail, newnode); // success or not both ok
	return flag;
}

__device__ int dequeue(pnode mynode, pqueue myqueue){
	pnode tail = NULL;
	pnode head = NULL;
	pnode next = NULL;
	while(1){
		head = myqueue->head;
		tail = myqueue->tail;
		next = head->next;
		//printf("In:dequeue\n");
		if(head == myqueue->head){
			if(head == tail){
				//printf("In:head == tail\n");
				if(next == NULL){
					//printf("Block:%d Thread:%d out:NULL\n", blockIdx.x, threadIdx.x);
					return -1;
				}
				else
					myAtomicCAS(&myqueue->tail, tail, next); // just try to do that...
			}
			else{
				//printf("In:head!=tail\n");
				mynode->data = next->data;
				if(head == myAtomicCAS(&myqueue->head, head, next)){
					break;
				}
			}
		}
	}

	//TODO first we don't delete node
	//deleteNode(head);

	return 0;
}

__device__ void deleteNode(pnode delnode){
	free(delnode);//TODO:delete node use memory reclamation
}
