#ifndef STRUCTURES_H
#define STRUCTURES_H 1
#include<vector>
#include "definitions.h"
#include "myvectors.h"

#include "cache.h"


//structure for storing and manipulating a multi-label problem
class M3LProblem
{
	public:

	//type of an example
	struct example_t
	{
		example_t(const SparseVector& x, const FullVector& y)
		 {
			
			inpt=x;
			
			labels=y;
			
		}
		example_t() {}
		SparseVector inpt;
		FullVector labels;
	};

	//vector to store all the examples
	typedef std::vector< example_t>  examples_t;
	examples_t data;
	
	//Number of examples
	int N; 
	
	//Maximum dimensionality of the feature vector
	int d;
	
	//Number of labels
	int L;

	//Load from a file in libSVM format
	int libsvm_load_data(char *filename);
		
	
	//Print labels in LibSVM format
	void print_labels(FullVector y, std::ostream& fout)
	{
		bool flag=false;
		for(int i=0;i<y.size();i++)
		{
			if(y.get(i)>0)
			{
			if(flag) fout<<",";
			fout<<i;
			flag=true;
			}
		}
		
	}
	
	//Print entire problem out to a file in LibSVM format
	void print_libsvm(std::ostream& fout)
	{
		for(int i=0;i<data.size();i++)
		{
			if(dot(data[i].inpt,data[i].inpt)>0)
			{
			print_labels(data[i].labels,fout);
			data[i].inpt.print_libsvm(fout);
			}
		}
	}
	
	
};


//Structure for Kernel parameters
struct KernelParameters
{
	int k_type;
	int degree;
	M3LFloat gamma;
	M3LFloat u_0;
};



//Structure for storing the learned model. It is more like a union of all the different kinds of models, be it linear or kernel
class M3LModel
{
	public:

	//support vector
	struct SupportVector
	{
		SparseVector x;
		FullVector theta;
		SupportVector(){}
		SupportVector(const FullVector& theta, const SparseVector& x):x(x),theta(theta){}
	};

	//number of labels
	int L;

	//type of the algo that generated this model: depending on the algo the model will be stored in different structures
	int AlgoType;

	//value of bias
	M3LFloat bias;

	//the matrix Z, used for LargeScale linear
	FullMatrix Zt;

	//list of support vectors, used for Kernel M3L
	std::vector<SupportVector> SVs;

	//Last column of Z, used for largescale linear
	FullVector b;

	//R matrix	
	FullMatrix R;

	//kernel parameters
	KernelParameters kparams;

};




//class containing paramters to the different algorithms
class M3LParameters
{
	public:
	M3LFloat C;
	M3LFloat tau;
	M3LFloat bias;
	int alg_type;
	int verbosity;
	M3LFloat cache_size;
	KernelParameters kparams;
};

//class for storing the testing results
class M3LTestingStats
{
	public:

	//Total Hamming Loss
	M3LFloat HammingLoss;

	//Hamming Loss for each individual label
	M3LFloat* HammingLossInd;

	//0-1 Loss
	M3LFloat ZeroOneLoss;

	//number of labels
	int L;

	//deallocate memory
	void destroy()
	{
		delete[] HammingLossInd;
	}
};


//Generic class for all learning algorithms
class M3LLearner
{
	public:
	//initialize training
	virtual void initialize(M3LProblem* problem)=0;

	//train
	virtual M3LModel* learn(M3LProblem* problem, M3LParameters* params)=0;
	
	//set R
	virtual void setR(FullMatrix R2)=0;

	//deallocate memory
	virtual void destroy()=0;
};

//Create appropriate learner given the parameters
M3LLearner* M3LCreateLearner(M3LParameters* param);

//Umbrella functions for learning
M3LModel* M3LLearn(M3LProblem* problem, M3LParameters* param);
M3LModel* M3LLearn(M3LProblem* problem, M3LParameters* param, FullMatrix R);

//Read model from file
M3LModel* M3LReadModel(char* input_file);

//Save model
void M3LSaveModel(M3LModel* model, char* model_file);



//Helper functions that compute Hamming Loss
inline M3LFloat compute_hamming_loss(M3LFloat y1, M3LFloat y2)
{
	
	return (y1==y2)?0:1;
}
M3LFloat compute_hamming_loss(FullVector ytrue, FullVector y);

//Test and print results
M3LTestingStats* M3LTest(M3LModel* model, M3LProblem* problem, char* output_file);
void M3LPrintTestingStats(M3LTestingStats* test);

//Helper functions for predicting label sets of a single point
FullVector predict(M3LModel* model, SparseVector x);
FullVector score(M3LModel* model, SparseVector x);

//dot product
inline M3LFloat kdot(const SparseVector& x, const SparseVector& y, const KernelParameters& kparams, const M3LFloat& bias=0)
{
	M3LFloat d;
		switch(kparams.k_type)
		{
		case K_LIN: return dot(x,y)+bias*bias;
			
		case K_RBF: d=dot(x,x) +dot(y,y) -2*dot(x,y);
			
			return exp(-kparams.gamma*d) + bias*bias;
		case K_POLY: d=dot(x,y);
			return pow(kparams.gamma*d+kparams.u_0, kparams.degree) + bias*bias;
		default: return 0;
		}
}

inline M3LFloat lineardot(const SparseVector& x, const SparseVector& y, const M3LFloat& bias=0)
{
	return dot(x,y)+bias*bias;
}
void M3LScaleToUnitNorm(M3LProblem* train, M3LProblem* test);
#endif
