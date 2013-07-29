//head always point to the flag one
//tail always point to last or last second one
//init:head = tail = flag one 
//malloc and free in enqueue && dequeue
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

using namespace std;

typedef struct node{
	//TODO any data type
	int data;
	struct node * next;
}node, * pnode;

typedef struct queue{
	pnode head;
	pnode tail;
}queue, *pqueue;

//__host__ void init(pqueue myqueue);

__device__ void enqueue(int mydata,pqueue myqueue);
__device__ int dequeue(pnode mynode, pqueue myqueue);
__device__ pnode myAtomicCAS(pnode * address, pnode compare, pnode val);
__device__ void deleteNode(pnode delnode);

//__global__ void app_bfs(pqueue myqueue, pnode d_dummy); // TODO add bfs to test queue.
__global__ void app_bfs(pqueue myqueue); // TODO add bfs to test queue.
__global__ void init(pqueue myqueue);
__global__ void show(pqueue myqueue);


int main(int argc, char * argv[]){
	int num_block, num_thread_perblock;
	
	//init and copy
	pnode h_dummy;
	pnode d_dummy;
	pqueue d_myqueue;

	if(argc != 3){
		printf("Usage: queue block_num thread_num\n");
		exit(1);
	}
	num_block = atoi(argv[1]);
	num_thread_perblock = atoi(argv[2]);

	h_dummy = (pnode)malloc(sizeof(node));
	h_dummy->data = -1;
	h_dummy->next = NULL;

	cudaMalloc((void **)&d_dummy, sizeof(node));
	cudaMalloc((void **)&d_myqueue, sizeof(queue));
	cudaMemcpy(d_dummy, h_dummy, sizeof(node), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
 	float elapsedTime;
 	cudaEventCreate(&start);
 	cudaEventCreate(&stop);
 	cudaEventRecord(start, 0);

	init<<<1,1>>>(d_myqueue);
	app_bfs<<<num_block,num_thread_perblock>>>(d_myqueue);
	show<<<1,1>>>(d_myqueue);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("[Info]Block:%d\tThread:%d\tElapsedTime:%fms\n", num_block, num_thread_perblock, elapsedTime); 

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(h_dummy);
	cudaFree(d_dummy);
	cudaFree(d_myqueue);
	//cudaDeviceSynchronize();
	printf("[Info]%s\n",cudaGetErrorString(cudaGetLastError()));

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

__global__ void show(pqueue myqueue){
   pnode temp = myqueue->head;
   while(temp != NULL){
   	printf("%d\t",temp->data);
   	temp = temp->next;
   }
   printf("\n");
	
}

__global__ void app_bfs(pqueue myqueue){
	//printf("%d:%d\n", threadIdx.x, d_dummy->data);
/*
	if((blockIdx.x == 0) && ( threadIdx.x == 0)){
		pnode d_dummy = (pnode)malloc(sizeof(node));
		d_dummy->data = -1;
		d_dummy->next = NULL;
		myqueue->head = d_dummy;
		myqueue->tail = d_dummy;
	}
	__syncthreads();
*/
	//printf("[Info:start]block%d:thread:%d\n", blockIdx.x, threadIdx.x);
	pnode newnode = (pnode)malloc(sizeof(node));
	//enqueue(blockIdx.x * blockDim.x + threadIdx.x, myqueue);

	if(blockIdx.x % 2 == 1){
	//if(1){
		//printf("block:%d\tthread:%d\n", blockIdx.x, threadIdx.x);
		enqueue(blockIdx.x * blockDim.x + threadIdx.x, myqueue);
	}
	else{
		//printf("block:%d\tthread:%d\n", blockIdx.x, threadIdx.x);
		dequeue(newnode, myqueue);
		
//		if (!dequeue(newnode, myqueue))
//			printf("Block:%d Thread:%d out:%d\n", blockIdx.x, threadIdx.x, newnode->data);
//		else
//			printf("Block:%d Thread:%d out:NULL\n", blockIdx.x, threadIdx.x);
	    
	}

/*
	__syncthreads();
	if((blockIdx.x == 0) && ( threadIdx.x == 0)){
    	pnode temp = myqueue->head;
    	while(temp != NULL){
    		printf("%d\t",temp->data);
    		temp = temp->next;
    	}
		printf("\n");
	}
*/
	//printf("[Info:end]block%d:thread:%d\n", blockIdx.x, threadIdx.x);
}

__device__ pnode myAtomicCAS(pnode * address, pnode compare, pnode val){
	return (pnode)atomicCAS((unsigned long long int*)address,
			(unsigned long long int)compare,
			(unsigned long long int)val);
}
/*
//TODO in host
__host__ void init(pqueue myqueue){
	pnode mynode = new node();
	mynode->data = 0;
	mynode->next = NULL;
	myqueue->head = myqueue->head = mynode;
}
*/

//__device__ void enqueue(pnode newnode,pqueue myqueue){
__device__ void enqueue(int newdata,pqueue myqueue){
	int count = 0;
	pnode tail = NULL,next = NULL;
	pnode newnode = (pnode)malloc(sizeof(node));
/*
   if (newnode == NULL){// added can avoid the unspecified launch failure!!!
		printf("[Error]Malloc failed!\n");
		return ;
	}
*/
	newnode->data = newdata;
	newnode->next = NULL;
	//printf("In:enqueue:%d\n", threadIdx.x);
	
	while(1){
		tail = myqueue->tail;
		next = tail->next;
		if(tail == myqueue->tail){
			//printf("%dIn:tail==queue_tail\n",threadIdx.x);
			if(next == NULL){
				//printf("%dnext==NULL\n", threadIdx.x);
				if(next == myAtomicCAS(&myqueue->tail->next, next, newnode)){
					printf("Block:%d Thread:%d in:%d\n", blockIdx.x, threadIdx.x, newnode->data);
					break;
				}
			}
			else{ 
				//printf("%dnext != NULL\n",threadIdx.x);
				myAtomicCAS(&myqueue->tail, tail, next);// success or not both ok
			}
		}
	}
	myAtomicCAS(&myqueue->tail, tail, newnode); // success or not both ok
	
	//printf("Out:enqueue\n");
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
					printf("Block:%d Thread:%d out:NULL\n", blockIdx.x, threadIdx.x);
					return -1;
				}
				else
					myAtomicCAS(&myqueue->tail, tail, next); // just try to do that...
			}
			else{
				//printf("In:head!=tail\n");
				mynode->data = next->data;
				if(head == myAtomicCAS(&myqueue->head, head, next)){
					//printf("out:%d\n",mynode->data);
					printf("Block:%d Thread:%d out:%d\n", blockIdx.x, threadIdx.x, mynode->data);
					break;
				}
			}
		}
	}

	//TODO first we dont delete node
	//deleteNode(head);

	return 0;
}

__device__ void deleteNode(pnode delnode){
	free(delnode);//TODO:delete node use memory reclamation
}
