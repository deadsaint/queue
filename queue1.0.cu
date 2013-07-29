//head always point to the flag one
//tail always point to last or last second one
//init:head = tail = flag one 
//malloc and free beyond enqueue && dequeue
#include <iostream>
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

//__device__ void init(pqueue myqueue);
__device__ void enqueue(int mydata,pqueue myqueue);
__device__ void dequeue(pnode mynode, pqueue myqueue);
__device__ pnode myAtomicCAS(pnode * address, pnode compare, pnode val);

int main(){
	//InitQueue
	return 0;
}


__device__ pnode myAtomicCAS(pnode * address, pnode compare, pnode val){
	return (pnode)atomicCAS((unsigned long long int*)address,
			(unsigned long long int)compare,
			(unsigned long long int)val);
}
/*
__device__ void init(pqueue myqueue){
	pnode mynode = new node();
	mynode->data = 0;
	mynode->next = NULL;
	myqueue->head = myqueue->head = mynode;
}
*/
__device__ void enqueue(pnode newnode,pqueue myqueue){
	pnode tail = NULL,next = NULL;
	while(1){
		tail = myqueue->tail;
		next = tail->next;
		if(tail == myqueue->tail){
			if(next == NULL)
				if(next == myAtomicCAS(&myqueue->tail->next, next, newnode))
					break;
			else 
				myAtomicCAS(&myqueue->tail, tail, next);// success or not both ok
		}
	}
	myAtomicCAS(&myqueue->tail, tail, newnode); // success or not both ok
}

__device__ void dequeue(pnode mynode, pqueue myqueue){
	pnode tail = NULL;
	pnode head = NULL;
	pnode next = NULL;
	while(1){
		head = myqueue->head;
		tail = myqueue->tail;
		next = head->next;
		if(head == myqueue->head){
			if(head == tail){
				if(next == NULL)
					return ;
				else
					myAtomicCAS(&myqueue->tail, tail, next); // just try to do that...
			}
			else{
				mynode = head;
				if(head == myAtomicCAS(&myqueue->head, head, next))
					break;
			}
		}
	}
//	delete head;
}
