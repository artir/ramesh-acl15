#include "structures.h"
#include "M3LKernel.h"

#include "M3LLinear.h"
#include <fstream>
#include<iostream>
int M3LProblem::libsvm_load_data(char *filename) 
{
	std::cout<<"Loading data\n";
	std::ifstream fin(filename);
	if(!fin.good())
	{
		std::cout<<"Couldn't find file.\n";
		exit(0);
	}
	//First find out the number of labels
	char c;
	int label;
	int max_label=0;
	int num=0;
	while(fin.good())
	{
		c=fin.get();
		if(c!=' ')
		{
			fin.unget();
			do
			{
				fin>>label;
				if(label>max_label) max_label=label;
			}while(fin.get()==',');
			
		}
		while(fin.get()!='\n' && fin.good()){}
		num++;	
		
	}
	data.reserve(num);
	L=max_label+1;
	d=0;
	fin.clear();
	fin.seekg(0, std::ios::beg);
	int count=0;
	fin.get();
	while(fin.good())
	{
		fin.unget();
		//FullVector y;
		data.push_back(example_t());
		data[count].labels.create(L,-1);
		
		SparseVector x;
		c=fin.get();
		if(c!=' ')
		{
			fin.unget();
			do
			{
				fin>>label;
				data[count].labels.set(label,1);
			}while(fin.get()==',');
			
		}
		
		
		
		data[count].inpt.read_libsvm(fin);
		d=mymax(d, data[count].inpt.last());
		num++;	
		count++;
		fin.get();
	}
	d++;
	N=data.size();
	std::cout<<"Data loaded\n";
	std::cout<<N<<" examples, "<<L<<" labels, "<<d<<" features\n";
	
}

M3LLearner* M3LCreateLearner(M3LParameters* param)
{
	
	switch(param->alg_type)
	{
		case ALG_KER: return new M3LKernel(param);
		
		case ALG_LIN:return new M3LLinear(param);
		default: return 0;
	}
}

M3LModel* M3LLearn(M3LProblem* problem, M3LParameters* param)
{
	std::cout<<"Training started..\n";
	M3LLearner* learner=M3LCreateLearner(param);
	learner->initialize(problem);
	M3LModel* model=learner->learn(problem, param);
	learner->destroy();
	delete learner;
	return model;
	
	



}
M3LModel* M3LLearn(M3LProblem* problem, M3LParameters* param, FullMatrix R)
{
	std::cout<<"Training started..\n";
	M3LLearner* learner=M3LCreateLearner(param);
	learner->initialize(problem);
	learner->setR(R);
	M3LModel* model=learner->learn(problem, param);
	learner->destroy();
	delete learner;
	return model;
	
	



}
void print_error_model(char* expected, char* found)
{
	fprintf(stderr, "Model file wrong! Expected: \"%s\", Found: \"%s\" \n", expected, found);
}


M3LModel* M3LReadModel(char* model_file)
{
	M3LModel* model=new M3LModel;
	std::ifstream fin(model_file);
	std::cout<<"Loading model\n";
	char tmp[100];
	char tmp2[100];
	M3LFloat theta;
	int N;
	if(!fin.good())
	{
		fprintf(stderr, "cannot open model file!\n");
		exit(-1);
	}
	fin>>tmp;
	if(strcmp(tmp, "#Kernelized")==0)
	{
		model->AlgoType = ALG_KER;
		
	}
	else if(strcmp(tmp, "#Linear")==0)
	{
		model->AlgoType = ALG_LIN;
	}
	
	switch(model->AlgoType)
	{
		case ALG_KER: fin>>tmp2;
				if(strcmp(tmp2, "#bias")!=0)
				{
					
					print_error_model("#bias",tmp2);
					exit(-1);
				}
				fin>>model->bias;
				fin>>tmp2;
				if(strcmp(tmp2, "#labels")!=0)
				{
					print_error_model("#labels", tmp2);
					exit(-1);
				}
				fin>>model->L;
				fin>>tmp2;
				if(strcmp(tmp2, "#kernel")!=0)
				{
					print_error_model("#kernel", tmp2);
					exit(-1);
				}
				fin>>model->kparams.k_type;
				if(model->kparams.k_type==K_RBF || model->kparams.k_type==K_POLY)
				{
					fin>>model->kparams.gamma;
					
				}
				if(model->kparams.k_type==K_POLY)
				{
					fin>>model->kparams.u_0;
				}
				if(model->kparams.k_type==K_POLY)
				{
					fin>>model->kparams.degree;
				}
				
				fin>>N;
				model->SVs.reserve(N);
				fin.get();
				while(fin.good())
				{
					FullVector Theta;
					SparseVector x;
					fin.unget();
					Theta.create(model->L, 0);
					for(int i=0;i<model->L;i++)
					{
						fin>>theta;
						Theta.set(i,theta);
					}
					
					x.read_libsvm(fin);
					model->SVs.push_back(M3LModel::SupportVector(Theta,x));
					fin.get();
				}
				fin.close();
				std::cout<<"Model loaded\n";
				return model;
		case ALG_LIN:	fin>>tmp2;
				if(strcmp(tmp2, "#bias")!=0)
				{
					print_error_model("#bias", tmp2);
					exit(-1);
				}
				fin>>model->bias;
				fin>>tmp2;
				if(strcmp(tmp2, "#labels")!=0)
				{
					print_error_model("#labels", tmp2);
					exit(-1);
				}
				fin>>model->L;
				fin>>tmp2;
				if(strcmp(tmp2, "#Z")!=0)
				{
					print_error_model("#Z", tmp2);
					exit(-1);
				}
				fin.get();
				fin>>model->Zt;
				fin>>tmp2;
				if(strcmp(tmp2, "#b")!=0)
				{
					print_error_model("#b", tmp2);
					exit(-1);
				}
				fin.get();
				fin>>model->b;
				
				fin.close();
				std::cout<<"Model loaded\n";
				return model;
		default: return 0;
	}
	
}




void M3LSaveModel(M3LModel* model, char* model_file)
{
	std::cout<<"Saving model\n";
	if(model==0)
	{
		fprintf(stderr, "Model is NULL!\n");
		exit(-1);
	}
	std::ofstream fout(model_file);
	switch(model->AlgoType)
	{
		case ALG_KER: fout<<"#Kernelized\n";
				fout<<"#bias "<<model->bias<<"\n";
				fout<<"#labels "<<model->L<<"\n";
				fout<<"#kernel "<<model->kparams.k_type<<"\n";
				if(model->kparams.k_type==K_RBF || model->kparams.k_type==K_POLY)
					fout<<model->kparams.gamma<<"\n";
				if(model->kparams.k_type==K_POLY)
					fout<<model->kparams.u_0<<"\n";
				if(model->kparams.k_type==K_POLY)
					fout<<model->kparams.degree<<"\n";
				
				fout<<model->SVs.size()<<"\n";
				
				for(int i=0;i<model->SVs.size();i++)
				{
					for(int j=0;j<model->L;j++)
					{
						fout<<model->SVs[i].theta.get(j)<<" ";
					}
					
					model->SVs[i].x.print_libsvm(fout);
					
				}
				
				break;
		case ALG_LIN:fout<<"#Linear\n";
				fout<<"#bias "<<model->bias<<"\n";
				fout<<"#labels "<<model->L<<"\n";
				fout<<"#Z\n";
				fout<<model->Zt<<"\n";
				fout<<"#b\n";
				fout<<model->b<<"\n";
				break;
		
		default: break;
	}
	fout.close();
	std::cout<<"Model saved\n";
	
}


void M3LPrintTestingStats(M3LTestingStats* test)
{
	int i;
	if(test==0)
	{
		fprintf(stderr,"Test Stats empty\n");
		exit(-1);
	}
	for(i=0;i<test->L;i++)
	{
		printf("Hamming Loss for label %d : %f%\n", i, test->HammingLossInd[i]*100.f);
	}
	printf("Net Hamming Loss: %f%\n", test->HammingLoss*100.f);
	
}

FullVector predict(M3LModel* model, SparseVector x)
{
	FullVector y;
	
	int i;
	M3LFloat ker;
	
	switch(model->AlgoType)
	{
		case ALG_KER: y.create(model->L,0);
				for(i=0;i<model->SVs.size();i++)
				{
					ker=kdot(x, model->SVs[i].x, model->kparams,model->bias);
					y.add(model->SVs[i].theta,ker);
					
				}
				for(i=0;i<model->L;i++)
					if(y.get(i)>0) y.set(i,1);
					else y.set(i,-1);
				break;
		case ALG_LIN:
				
				
				y=mult(model->Zt, x);
				for(i=0;i<model->L;i++)
				{
					M3LFloat tmp=y.get(i);
					tmp+=model->bias*model->b.get(i);
					y.set(i, tmp>0?1:-1);
				}	
				break;
		
		default:break;
	}
	return y;
}
FullVector score(M3LModel* model, SparseVector x)
{
	
	FullVector y;
	
	int i;
	M3LFloat ker;
	
	switch(model->AlgoType)
	{
		case ALG_KER: y.create(model->L,0);
				for(i=0;i<model->SVs.size();i++)
				{
					ker=kdot(x, model->SVs[i].x, model->kparams,model->bias);
					y.add(model->SVs[i].theta,ker);
					
				}
				break;
		case ALG_LIN:
				
				
				y=mult(model->Zt, x);
				for(i=0;i<model->L;i++)
				{
					M3LFloat tmp=y.get(i);
					tmp+=model->bias*model->b.get(i);
					y.set(i, tmp);
				}	
				break;
		
		default:break;
	}
	return y;
}
M3LFloat compute_hamming_loss(FullVector ytrue, FullVector y)
{
	M3LFloat HL=0;
	int i;
	for(i=0;i<ytrue.size();i++)
	{
		if(ytrue.get(i)!=y.get(i)) HL++;
	}
	return HL/ytrue.size();
}
M3LFloat compute_zero_one_loss(FullVector ytrue, FullVector y)
{
	M3LFloat flag=0;
	for(int i=0;i<ytrue.size();i++)
	{
	      if(ytrue.get(i)!=y.get(i)) 
	      {
		 flag = 1;
	      }
	}
	return flag;

}
M3LTestingStats* M3LTest(M3LModel* model, M3LProblem* problem, char* output_file)
{
	
	std::cout<<"Testing started..\n";
	std::ofstream foutest(output_file);
	int i, j;
	FullVector y;
	M3LFloat sz=(M3LFloat)problem->data.size();
	M3LTestingStats* testingstats = new M3LTestingStats();
	testingstats->HammingLoss=0;
	testingstats->HammingLossInd=new M3LFloat[model->L];
	for(j=0;j<model->L;j++)
	{
		testingstats->HammingLossInd[j]=0;
	}
	testingstats->L=model->L;
	
	for(i=0;i<sz;i++)
	{
		if(i%1000==0)
		{
			printf(".");
			fflush(stdout);
		}
		y=predict(model, problem->data[i].inpt);
		foutest<<y;
		testingstats->HammingLoss+=compute_hamming_loss(problem->data[i].labels, y)/sz;
		testingstats->ZeroOneLoss+=compute_zero_one_loss(problem->data[i].labels, y)/sz;
		
		for(j=0;j<model->L;j++)
		{
			testingstats->HammingLossInd[j]+=compute_hamming_loss(problem->data[i].labels.get(j), y.get(j))/sz;	
		}
		y.clear();
	}
	printf("\n");
	foutest.close();
	return testingstats;
}









