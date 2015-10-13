
#include "structures.h"
#if WINDOWS
bool time_started = false;
time_t _start, _finish;
#endif
FullMatrix R;
bool Rthere;
M3LProblem* read_problem(char* input_file)
{
	M3LProblem* problem=new M3LProblem;
	problem->libsvm_load_data(input_file);
	return problem;
}
void set_default_parameters(M3LParameters* params)
{
	params->C=0.5;
	params->tau=0.001;
	params->bias=1;
	params->kparams.gamma=1;
	params->kparams.k_type=K_RBF;
	params->kparams.degree=3;
	params->kparams.u_0=1;
	params->alg_type = ALG_KER;
	params->cache_size=1000.0;
	params->verbosity=1;
	Rthere=false;
}





void exit_with_help()
{
	M3LParameters params;
	set_default_parameters(&params);
	std::cout<<"Usage:\n";
	std::cout<<"Training: The command\n";
	std::cout<<"\tM3L -train [options] training_file model_file\n";
	std::cout<<"reads in training data from training_file and saves the learnt M3L classifier in model_file subject to the following options:\n";
	
	std::cout<<"-C misclassification penalty. Default: "<<params.C<<"\n";
	std::cout<<"-a optimization algorithm. \n";
	std::cout<<"\t0: Kernelized -- SMO";
	if(params.alg_type==0) std::cout<<"(Default)\n";
	else std::cout<<"\n";
	std::cout<<"\t1: Linear -- Dual co-ordinate ascent with shrinkage (choose this if you have a linear kernel)";
	if(params.alg_type==1) std::cout<<"(Default)\n";
	else std::cout<<"\n";
	
	std::cout<<"-t tau. Stopping parameter. Algorithm stops when all projected gradients have magnitude less than tau. Default: "<<params.tau<<"\n";
	std::cout<<"-k kernel_type (valid only with -a 0). \n";
	std::cout<<"\t 0: Linear: x'*y";
	if(params.kparams.k_type==0) std::cout<<"(Default)\n";
	else std::cout<<"\n";
	std::cout<<"\t 1: RBF: exp(-gamma*||x-y||^2)";
	if(params.kparams.k_type==1) std::cout<<"(Default)\n";
	else std::cout<<"\n";
	std::cout<<"\t 2: Poly: (gamma*x'*y+u_0)^degree";
	if(params.kparams.k_type==2) std::cout<<"(Default)\n";
	else std::cout<<"\n";
	
	std::cout<<"-g gamma(kernel parameter for RBF and polynomial kernels). Default "<<params.kparams.gamma<<"\n";
	std::cout<<"-d degree. Default "<<params.kparams.degree<<"\n";
	std::cout<<"-r u_0. Default "<<params.kparams.u_0<<"\n";
	std::cout<<"-b bias. Default "<<params.bias<<"\n";
	std::cout<<"-m cache size in MB. Default "<<params.cache_size<<" (the best results are obtained when the cache is large enough to hold the entire kernel matrix)\n";
	std::cout<<"-R correlation matrix file. Look at example_R.txt for an example correlation matrix file (Defaults to using the identity matrix).\n";
	std::cout<<"\n\nTesting: The command\n";
	std::cout<<"\tM3L -test testing_file model_file output_file\n"; 
	std::cout<<"applies the learnt M3L classifier in model_file to the test data in testing_file and saves the predictions in output_file.\n";
	std::cout<<"\n\nThe training, testing, model and output prediction files all follow the LIBSVM file format. \n";
}
void parse_command_line(int argc, char** argv, M3LParameters &parameters, char* input_file, char* model_file)
{
	set_default_parameters(&parameters);
	std::ifstream fin;
	int i;
	for(i=1;i<argc;i++)
	{
	
  		if(argv[i][0]!='-') 
    			break;
  		switch(argv[i][1])
  		{
    			case 'C': 	parameters.C=atof(argv[i+1]);
	      				break;
    			case 't': 	parameters.tau=atof(argv[i+1]);
	      				break;
    			case 'k': 	parameters.kparams.k_type=atoi(argv[i+1]);
	      				break;
    			case 'b': 	parameters.bias=atof(argv[i+1]);
					break;
    			case 'm': 	parameters.cache_size=atof(argv[i+1]);
	      				break;
    			case 'g': 	parameters.kparams.gamma=atof(argv[i+1]);
					break;
			case 'd':	parameters.kparams.degree=atoi(argv[i+1]);
					break;
			case 'r':	parameters.kparams.u_0=atof(argv[i+1]);
					break;
    			case 'R': 	Rthere=true;
					fin.open(argv[i+1]);
					if(!fin.good())
					{
						fprintf(stderr, "Cannot open %s\n", argv[i+1]);
						exit(-1);
					}
					fin>>R;
					fin.close();
					break;
    			case 'a': 	parameters.alg_type=atoi(argv[i+1]);
					break;
    			case 'v': 	parameters.verbosity=atoi(argv[i+1]);
					break;
    			default: 	fprintf(stderr, "unknown option %s\n", argv[i]);
	      				break;
  		}
  		i++;
	}
	if(i>=argc-1)
	{
		fprintf(stderr, "Not enough arguments!\n");
		exit(-1);
	}
	strcpy(input_file,argv[i]);
	strcpy(model_file,argv[i+1]);
	
}

int learn_main(int argc, char** argv)
{
  	M3LParameters parameters;
  	char input_file_name[255];
  	char model_file_name[255];
 	parse_command_line(argc, argv,parameters,input_file_name, model_file_name);
  	M3LProblem* problem = read_problem(input_file_name);
	M3LModel* model;
  	if(Rthere)
		model=M3LLearn(problem, &parameters, R);
  	else
  		model=M3LLearn(problem, &parameters);
  	M3LSaveModel(model,model_file_name);
  	return 0;

}
int classify_main(int argc, char** argv)
{
	if(argc<=3) 
	{
		fprintf(stderr, "Not enough arguments!\n");
		exit(-1);
	}
	char* input_file=argv[1];
	char* model_file=argv[2];
	char* output_file=argv[3];
	M3LProblem* problem = read_problem(input_file);
	M3LModel* model=M3LReadModel(model_file);
	M3LTestingStats* test_stats=M3LTest(model, problem, output_file);
	M3LPrintTestingStats(test_stats);
	return 0;
}

int main(int argc, char** argv)
{
	if(argc<2)
  	{
    		fprintf(stderr, "Not enough arguments!\n");
    		exit_with_help();
    		return 0;
  	}
  	bool train=strcmp(argv[1],"-train")==0;
  	bool test=strcmp(argv[1],"-test")==0;
	if(!train && !test)
	{
		fprintf(stderr, "Invalid commandline argument:%s\n", argv[1]);
		exit_with_help();
   		return 0;
	}
	argc--;
	argv++;
	if(train)
	{
		return learn_main(argc, argv);
	}
	if(test)
	{
		return classify_main(argc, argv);
	}
}
