#include "M3LLinear.h"
#include "definitions.h"
#include "myvectors.h"
#include "cache.h"
#define INF FLT_MAX
#define max mymax
#define min mymin

//helper functions
void add(M3LFloat* s, SparseVector a, M3LFloat scale)
{
	
	int nnz=a.get_nnz();
	for(int i=0;i<nnz;i++)
	{
		s[a.get_ith_index(i)]+=scale*(a.get_ith_value(i));
	}
}
void clear(M3LFloat* s, svm_node* start, int cnt)
{
	svm_node* p=start;
	int c=0;
	while(p!=0 && c<cnt)
	{
		c++;
		s[p->index]=0;
		p++;
	}
}
int fill(M3LFloat* s, svm_node* start, int last)
{
	svm_node* p=start;
	int i=0;
	int cnt=0;
	while(i<=last && p!=0)
	{
		
		if(s[i]!=0)
		{
			p->index=i;
			p->value=s[i];
			p++;
			cnt++;
		}
		i++;
		
	}
	return cnt;
	
}




M3LFloat M3LLinear::score(const int& l, SparseVector& x)
{
	M3LFloat* Zl=getZrow(l);
	int nnz=x.get_nnz();
	M3LFloat sum=0;
	
	for(int i=0;i<nnz;i++)
	{
		sum+=Zl[x.get_ith_index(i)]*x.get_ith_value(i);
		
	}
	sum+=bias*b[l];
	
	return sum;
}

//update only current row of Z^t
void M3LLinear::update_current_row(const int& l, SparseVector& x, const M3LFloat& scale)
{
	M3LFloat* Zl=getZrow(l);
	int nnz=x.get_nnz();
	for(int i=0;i<nnz;i++)
	{
		Zl[x.get_ith_index(i)]+=scale*(x.get_ith_value(i));
	}
	b[l]+=scale*bias;
}

//update all rows of Z^t
void M3LLinear::update_all_rows( SparseVector& x, const M3LFloat* Rrow, const M3LFloat& scale)
{
	
	
	M3LFloat* Zk=Z;
	M3LFloat mk;
	int nnz=x.get_nnz();
	for(int k=0;k<L;k++)
	{
		
		if(Rrow[k]==0)
		{
			Zk+=d;
			continue;
		}
		mk=Rrow[k]*scale;
		for(int i=0;i<nnz;i++)
		{
			Zk[x.get_ith_index(i)]+=mk*(x.get_ith_value(i));
		}
		b[k]+=Rrow[k]*scale*bias;	
		Zk+=d;
	}
	
}

//update other rows of Z^t
void M3LLinear::update_other_rows(const int& l, svm_node* start, const M3LFloat* Rrow, const M3LFloat& scale, const M3LFloat& update, const int& cnt)
{
	
	
	M3LFloat* Zk=Z;
	M3LFloat kthupdate;
	M3LFloat lastcol=bias*update;
	
	for(int k=0;k<L;k++)
	{
		if(k==l || Rrow[k]==0)
		{
			Zk+=d;
			continue;
		}
		svm_node* p=start;
		kthupdate=Rrow[k]*scale;
		
		for(int c=0;c<cnt;c++)
		{
			
			Zk[p->index]+=kthupdate*p->value;
			p++;
		}
	
		b[k]+=kthupdate*lastcol;	
		Zk+=d;
	}
	
}




void M3LLinear::initialize(M3LProblem* problem)
{
	
	N=problem->N;
	d=problem->d;
	L=problem->L;
	prob=problem;
	R=FullMatrix::identity(L);
	int i;
	Z=new M3LFloat[L*d];
	
	b=new M3LFloat[L];
	K_ii=new M3LFloat[N];
	active_size=new int[L];
	ptindices=new int*[L];
	labelindices=new int[L];
	
	
	lb=new M3LFloat*[L];
	PG_min=new M3LFloat[L];
	PG_max=new M3LFloat[L];
	shrink_lb=new M3LFloat[L];
	shrink_ub=new M3LFloat[L];
	corr_PG_max=new M3LFloat[L];
	corr_PG_min=new M3LFloat[L];
	
	current_update=new M3LFloat[d];
	
	for(i=0;i<N;i++)
	{
		K_ii[i]=lineardot(prob->data[i].inpt, prob->data[i].inpt, bias);
	}
	for(i=0;i<L;i++)
	{
		b[i]=0;
		
		ptindices[i]=new int[N];
		labelindices[i]=i;
		lb[i]=new M3LFloat[N];
		for(int j=0;j<N;j++)
		{
			ptindices[i][j]=j;
			lb[i][j]=prob->data[j].labels.get(i)>0?0:-2*C;
		}
	}
	for(i=0;i<L*d;i++)
	{
		Z[i]=0;
	}
	
	for(i=0;i<d;i++)
	{
	  current_update[i]=0;
	}
	opt=new bool[L];
	finish_pass=new bool[L];
	index=new int[L];
	n=new svm_node[d];
	mode=new bool[L];
	
}




//the *BIG* function
void M3LLinear::train()
{
	
    	double starttime=getTime();
	M3LFloat dual=0;	
	int i;
	int update_called=0;
	
	//store all the labels in (easily accessible) arrays to avoid calling accessor functions of LaFVector
	const M3LFloat** labels=new const M3LFloat*[N];
	for(i=0;i<N;i++)
	{
		labels[i]=prob->data[i].labels.getData();
	}

	//initialize stuff
	M3LFloat taulimit=tau*10;
	for(i=0;i<L;i++)
	{
	  	shrink_ub[i]=FLT_MAX;
	  	shrink_lb[i]=-FLT_MAX;
	  	active_size[i] = N;
	  	opt[i]=false;
	  	corr_PG_max[i]=FLT_MAX;
	  	corr_PG_min[i]=-FLT_MAX;
	  	mode[i]=false;
	  
	}
	double PG;
	long long int iter=0;
	long long int totiter=0;
	
	
	bool	shrink=true;
	
	//start training
	while(1)
	{
	
		totiter++;
		
		
		//initialize iteration: initialize some variables and permute training examples
		//for each label
		for(i=0;i<L;i++)
		{
		    
			finish_pass[i]=false;
		    	index[i]=0;
			
			
		    	//if *possibly* optimal skip
		    	if(opt[i] ) 
		    	{
				continue;
		    	}

		    	PG_max[i]=-FLT_MAX;
		    	PG_min[i]=FLT_MAX;
		    	//permute training examples
		    	for (int k=0; k<active_size[i]; k++)
		    	{
				int j = k+rand()%(active_size[i]-k);
				swap_indices(i, k, j);
		    	}
		}
		//end for
		
		//start pass
		int num_passed=0;
		
		//permute labels
		for(int k=0;k<L;k++)
		{
		    	int l=k+rand()%(L-k);
		    	swap_labels(l,k);
		}
		
		//pass through the whole active set
		while(num_passed<L)
		{
		
		  	for(int k=0;k<L;k++)
		  	{
					
		    		i=labelindices[k];
				if(finish_pass[i]) continue;
		    
		    		//optimal - skip
		     		if(opt[i] && shrink) 
		    		{
					if(!finish_pass[i])
					{
			  			finish_pass[i]=true;
			  			num_passed++;
					}
					continue;
		    		}
		    
		    		M3LFloat update_total=0;
		    		bool finished=false;
		    		int maxindex=0;
					
		    		for(;index[i]<active_size[i] &&!finished;index[i]++)	
		    		{	
					
		  			if(iter%10000==0) 
		  			{
			    			printf(".");
			    			fflush(stdout);
		  			}
					int s=index[i];
					iter++;
					int labelk = i;
					int pti=ptindices[i][index[i]];	
					//compute gradient
					
					
					M3LFloat grad=labels[pti][i] - score(i, prob->data[pti].inpt);
					
					//compute lower and upper bounds
					M3LFloat lower=lb[labelk][pti];
					M3LFloat upper = lower+2.0*C;
		
					//compute PG
					PG=0;

					//if \alpha_ij is at its lower bound and grad<shrink_lb shrink
					//else if \alpha_ij is at its upper bound and grad>shrink_ub shrink
					if(lower==0)
					{
						if(grad<shrink_lb[labelk] && shrink)
						{
							active_size[i]--;
							swap_indices(i, active_size[i], s);
							if(index[i]==active_size[i])
							{
								finish_pass[i]=true;
						  		num_passed++;
							}
						
							index[i]--;
							continue;
						}
						else if(grad>0)
						{
							PG=grad;
						}
					}
					else if(upper==0)
					{
						if(grad>shrink_ub[labelk] && shrink)
						{
						
							active_size[i]--;
							swap_indices(i, active_size[i], s);
							if(index[i]==active_size[i])
							{
						  		finish_pass[i]=true;
						  		num_passed++;
							}
							index[i]--;
							continue;
						}
						else if(grad<0)
						{
							PG=grad;
						}
					}
					else
						PG=grad;
					PG_max[i]=max(PG_max[i], PG);
					PG_min[i]=min(PG_min[i], PG);	

					//if PG is nonzero solve
					if(fabs(PG) >= tau)
					{
					
						M3LFloat increase = grad/(K_ii[pti]*R.get(labelk, labelk));
						increase=min(max(increase, lower),upper);
						lb[labelk][pti]-=increase;
						
						//mode[i] dictates whether we are ready to do a sequence of updates along one label or whether we switch to another label immediately. In the initial stages, mode[i] is false and we switch to the next label immediately. Once the PGs fall below a certain threshold we switch to working along one label for some time before switching
						if(!mode[i])
						{

							//perform the full update
							update_all_rows(prob->data[pti].inpt, R.getRow(i), increase);
						}
						else
						{
					
							//update the row of Z^t corresponding to the current label. The other labels will be updated later
 							update_current_row(i, prob->data[pti].inpt, increase*R.get(i,i));
							maxindex=max(maxindex, prob->data[i].inpt.last());
 							update_total+=increase;

							//store some information so that we can update the other rwos of Z^t later
							add(current_update, prob->data[pti].inpt, increase);
						}
				
						dual+=-0.5*increase*increase*(K_ii[pti]*R.get(labelk, labelk))+increase*grad;
				
					}
		  			if(!mode[i]) finished=true;
		  		}
		
				if(mode[i])
				{				
 					//complete the update for other labels  
					int cntupdate=fill(current_update,n,maxindex);
					update_other_rows(i, n, R.getRow(i), 1, update_total,cntupdate);
 		  			clear(current_update, n, cntupdate);	
		 		}   
				if(index[i]==active_size[i])
		    		{
		      			finish_pass[i]=true;
		      			num_passed++;
		    		}
				
			}
		
		  
		  
		}
		int optcount=0;
		int passdcount=0;
		M3LFloat PGabsmax;
		//finish iteration: update shrink_lb and shrink_ub and check if optimum has been reached
		for(i=0;i<L;i++)
		{
			PGabsmax=max(fabs(PG_max[i]),fabs(PG_min[i]));
			if(PGabsmax<taulimit) mode[i]=true;
			
		  	//compute correlated PG max and correlated PG min
		  	corr_PG_max[i]=PG_max[i];
		  	corr_PG_min[i]=PG_min[i];
		  	for(int j=0;j<L;j++)
		  	{
		    		if(opt[j]) continue;
		    		if(R.get(i,j)==0) continue;
		    		corr_PG_max[i]=max(corr_PG_max[i], PG_max[j]*fabs(R.get(i,j)));
		    		corr_PG_min[i]=min(corr_PG_min[i], PG_min[j]*fabs(R.get(i,j)));
		  	}
			M3LFloat pa=max(0,PG_max[i]);
			M3LFloat pb=min(0, PG_min[i]);
		  	//if max |PG| < tau then that label reached its optimum the last time it was passed on its active set
		  	if(max(fabs(pa),fabs(pb)) < tau)
		  	{
				//if active size=N then optimum on the whole set
				//else optimum on smaller set: expand
				if(active_size[i] == N)
				{
					
					opt[i]=true;
					optcount++;
				}
				else
				{
					
					opt[i]=false;
					active_size[i] = N;
					shrink_ub[i] = FLT_MAX;
					shrink_lb[i] = -FLT_MAX;
					continue;
				}
		  	}
		  	else opt[i]=false;
				
		
			//update shrink_lb and shrink_ub
		  	shrink_ub[i] = corr_PG_max[i];
		  	shrink_lb[i] = corr_PG_min[i];
		  	if (shrink_ub[i] <= 0)
				shrink_ub[i] = INF;
		  	if (shrink_lb[i] >= 0)
				shrink_lb[i] = -INF;
		
		}
		
		
		if(optcount==L )
		{
		
			M3LFloat maxPGhere=0;
			//see if optimum has indeed been reached. If not, repeat
			int cnt=0;
		  	for(int i=0;i<L;i++)
	 	 	{
				for(int pti=0;pti<N;pti++)
				{
					int labelk=i;
					M3LFloat grad=prob->data[pti].labels.get(i) - score(i, prob->data[pti].inpt);
			
					//compute lower and upper bounds
					M3LFloat lower=lb[labelk][pti];
					M3LFloat upper = lower+2.0*C;
			
					//compute PG
					PG=0;
					if(lower==0)
					{
						if(grad>0) PG=grad;
					}
					else if(upper==0)
					{
						if(grad<0) PG=grad;
					}
					else PG=grad;
					if(fabs(PG)>tau) 
					{
						opt[i]=false;
						maxPGhere=mymax(maxPGhere, fabs(PG));
						cnt++;
					}
				}
		 	}
		  	if(cnt==0) break;
			else
			{
				printf("*");
				fflush(stdout);
				
				for(int i=0;i<L;i++)
				{
					opt[i]=false;
					mode[i]=false;
					
					fflush(stdout);
				
				}
				taulimit=taulimit/2.0;
			}
		}
		if(optcount==L)
		{
		  
		  	for(i=0;i<L;i++)
		  	{
		    		active_size[i] = N;
		    
				corr_PG_max[i]=FLT_MAX;
				corr_PG_min[i]=-FLT_MAX;
		    		shrink_ub[i] = FLT_MAX;
		    		shrink_lb[i] = -FLT_MAX;
		  	}
		}
	}	
	
	std::cout<<"\nTime = "<<getTime()-starttime<<"\n";
	std::cout<<"Iterations="<<iter<<"\n";
	
}
M3LModel* M3LLinear::learn(M3LProblem* problem, M3LParameters* params)
{

 	train();
	
	M3LModel* model=new M3LModel;
	model->L=L;
	model->kparams.k_type=K_LIN;
	model->AlgoType=ALG_LIN;
	model->bias=bias;
	model->Zt=FullMatrix::zero(L,d);
	model->b.create(L,0);
	int k=0;
	for(int i=0;i<L;i++)
	{
		for(int j=0;j<d;j++)
		{
			
			model->Zt.set(i,j,Z[k]);
			
			k++;
		}
		model->b.set(i, b[i]);
	}
	return model;
}
void M3LLinear::destroy()
{
	//clean up
	delete[] n;
	delete[] Z;
	delete[] b;
	delete[] K_ii;
	delete[] active_size;
	int i;
	for(i=0;i<L;i++)
	{
		delete[] ptindices[i];
		delete[] lb[i];
	}
	delete[] lb;
	delete[] ptindices;
	delete[] labelindices;
	delete[] PG_min;
	delete[] PG_max;
	delete[] shrink_lb;
	delete[] shrink_ub;
	delete[] corr_PG_max;
	delete[] corr_PG_min;
	delete[] current_update;
	delete[] opt;
	delete[] finish_pass;
	delete[] index;
	delete[] mode;
	
	
	
}
