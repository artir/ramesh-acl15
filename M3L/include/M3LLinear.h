#ifndef M3LLINEAR_H
#define M3LLINEAR_H
#include "structures.h"


//main learner class
class M3LLinear: public M3LLearner
{
	//Current Z^t, stored in row major order as one large float array of size L*d
	M3LFloat* Z;

	//Last column of Z^t: float array of size L
	M3LFloat* b;
	
	//Makes sense to precompute K_{ii} and store it for speed
	M3LFloat* K_ii;

	//Size of active set of each label
	int* active_size;

	//things needed to permute the training examples and labels cheaply
	int** ptindices;
	int* labelindices;

	//min and max PG encountered along a label
	M3LFloat* PG_min;
	M3LFloat* PG_max;

	//Bounds for shrinking
	M3LFloat* shrink_lb;
	M3LFloat* shrink_ub;

	
	M3LFloat* corr_PG_max;
	M3LFloat* corr_PG_min;
	M3LFloat* current_update;
	M3LFloat** lb;
	bool* opt;
	bool* finish_pass;
	int* index;
	svm_node* n;
	bool* mode;
	
	
	//The problem
	M3LProblem* prob;
	
	//number of pts
	int N;
	
	//number of labels
	int L;

	//dimensionality of the feature vector
	int d;

	//parameters
	M3LFloat C;
	M3LFloat C2;
	M3LFloat bias;
	M3LFloat tau;
	FullMatrix R;
	
	//get a row of Z^t
	M3LFloat* getZrow(int labelnum)
	{
		return &Z[labelnum*d];
	}


	//Helper functions for permuting examples
	void swap_indices(int labelnum, int i, int j)
	{
		int tmp=ptindices[labelnum][i];
		ptindices[labelnum][i]=ptindices[labelnum][j];
		ptindices[labelnum][j] = tmp;
		
	}
	void swap_labels(int k, int l)
	{
	  	int tmp=labelindices[k];
	  	labelindices[k]=labelindices[l];
	  	labelindices[l]=tmp;
	}

	//find the score of a particular point for a particular label
	M3LFloat score(const int& l,  SparseVector& x);


	//functions for updating after every step
	void update_current_row(const int& l, SparseVector& x, const M3LFloat& scale);
	void update_other_rows(const int& l, svm_node* start, const M3LFloat* Rrow, const M3LFloat& scale, const M3LFloat& update, const int& cnt);
	void update_all_rows( SparseVector& x, const M3LFloat* Rrow, const M3LFloat& scale);

	//The big function that does the training
	void train();
	
	public:

	
	M3LLinear(){}
	M3LLinear(M3LParameters* param)
	{
		C2=2.0*param->C;
		C=param->C;
		bias=param->bias;
		tau=param->tau;
		
		
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