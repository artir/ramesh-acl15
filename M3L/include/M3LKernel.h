#ifndef M3LKERNEL2_H
#define M3LKERNEL2_H 1
#include "structures.h"
 
class M3LKernel: public M3LLearner
{
	int num_cache_misses;

	//array to store all the gradients
	M3LFloat** gradients;

	//array to store all the theta
	M3LFloat** Theta;

	M3LFloat* maxR;
	M3LFloat maxgrad2;
	M3LFloat maxupdate;
	//indices pointing to the gradient in question
	int maxgradl;
	int maxgradi;

	//number of pts
	int N;

	//number of dims
	int d;

	//number of labels
	int L;

	//parameters
	M3LFloat C2;
	M3LFloat C;
	M3LFloat bias;
	M3LFloat tau;
	KernelParameters kparams;
	M3LFloat dual;
	M3LFloat cache_size;
	
	int freq;
	FullMatrix R;

	//pointer to the training examples and training labels
	SparseVector* train_examples;
	M3LProblem* prob;
	M3LFloat** train_labels;

	//Kernel cache
	Cache* cache;

	//Makes sense top precomputes the K_ii[]
	M3LFloat* K_ii;

	//arrays to help us do the update	
	M3LFloat* current_update;
	M3LFloat* PGs;
	
	
	//the *BIG* function
	void train();
	//get projected gradient given the gradient
	M3LFloat PG(M3LFloat g, M3LFloat theta)
	{
	  M3LFloat tmp=(g>0 && theta>=C2)?0:g;
	  tmp=(g<0 && theta<=0)?0:tmp;
	  return tmp;
	}
	//Some helper functions
	bool check_shift(int itn, M3LFloat dp)
	{
		
		bool b2=maxgradi==-1 || maxgradl==-1;
		b2=b2?b2:fabs(PG(gradients[maxgradl][maxgradi], Theta[maxgradl][maxgradi]))<0.05*(maxgrad2+maxupdate);
		return itn<2*L && !b2;
	}
	bool check_whole(int itn, M3LFloat dp)
	{
		return true;
	}
	void check_again();
	
	
	//get ith row of Kernel matrix
	M3LFloat* get_K(int i)
	{
		M3LFloat *data;
		int start;
		
		if((start = cache->get_data(i,&data,N)) < N)
		{
			
			num_cache_misses++;	
			for(int j=start;j<N;j++)
				data[j] = (M3LFloat)kdot(prob->data[i].inpt,prob->data[j].inpt, kparams, bias);
		}
		return data;
	}

	

	//optimize over two variables
	M3LFloat double_step();
	
	//optimize over single variable
	M3LFloat single_step(int p, int label);

	//update functions
	void update_label(int p, int label, M3LFloat delta);
	void update_double_label(int p, int q, int label, M3LFloat deltap, M3LFloat deltaq, M3LFloat* Kp);
	void update_all_gradients(int l);
	
	public:
	M3LKernel(){}
	M3LKernel(M3LParameters* param)
	{
		C2=2.0*param->C;
		C=param->C;
		bias=param->bias;
		tau=param->tau;
		cache_size=param->cache_size;
		kparams=param->kparams;
		
		
		
	}
	virtual void setR(FullMatrix R2)
	{
	  	R=R2;
	}
	virtual void initialize(M3LProblem* problem);
	
	virtual M3LModel* learn(M3LProblem* problem, M3LParameters* params);
	virtual void destroy();
	



};
#endif
