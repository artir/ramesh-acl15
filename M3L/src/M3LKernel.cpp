#include "M3LKernel.h"
#include "definitions.h"
#include "myvectors.h"
#include "cache.h"
#include<vector>
void M3LKernel::initialize(M3LProblem* problem)
{
	
	
	int i,j;
	N=problem->N;
	d=problem->d;
	L=problem->L;
	R=FullMatrix::identity(L);
	maxR=new M3LFloat[L];
	gradients=new M3LFloat*[L];
	Theta=new M3LFloat*[L];
	for(i=0;i<L;i++)
	{
		gradients[i]=new M3LFloat[N];
		Theta[i]=new M3LFloat[N];
	}
	prob=problem;
	
	train_labels=new M3LFloat*[N];
	current_update=new M3LFloat[N];
	K_ii=new M3LFloat[N];
	PGs=new M3LFloat[N];
	maxgradl=-1;
	maxgradi=-1;
	M3LFloat gmax=0;
	M3LFloat tmp;
	
 	for(i=0;i<N;i++)
	{
		current_update[i]=0;
		
		M3LFloat* y=problem->data[i].labels.getData();
		train_labels[i]=y;
	
		SparseVector& x=prob->data[i].inpt;
		K_ii[i]=kdot(x, x, kparams, bias);
	
		//LaFVector m=mult(R,problem->data[i].labels);
		
		for(j=0;j<L;j++)
		{
      
			gradients[j][i]=y[j];
			
			Theta[j][i]=train_labels[i][j]>0?0:2*C;
			tmp=(Theta[j][i]>=C2 && gradients[j][i]>=0)?0:gradients[j][i];
			tmp=(Theta[j][i]<=0 && gradients[j][i]<=0)?0:tmp;
			tmp=fabs(tmp);
			if(tmp>gmax && tmp>tau)
			{
			  	maxgradi=i;
			  	maxgradl=j;
			  	gmax=tmp;
		  
			}
		}
	}
	dual=0;
	cache=new Cache(N,cache_size*(1<<20));
	
	if(maxgradl==-1)
	{
		
		 return;
	}
	j=maxgradl;
	
	for(int i=0;i<N;i++)
	{
		tmp=(Theta[j][i]>=C2 && gradients[j][i]>=0)?0:gradients[j][i];
		tmp=(Theta[j][i]<=0 && gradients[j][i]<=0)?0:tmp;
		PGs[i]=fabs(tmp);
	}
	
}


//The *BIG* function
void M3LKernel::train()
{
	for(int i=0;i<L;i++)
	{
		maxR[i]=0;
		for(int j=0;j<L;j++)
		{
			if(i==j) continue;
			maxR[i]=mymax(maxR[i], fabs(R.get(i,j)));
		}
	}
	num_cache_misses=0;
	double start_time=getTime();
	int current_label;
	M3LFloat dualprogress;
	int itns=0;
	long long int totitns=0;
	bool full=false;
	do{
		//maxgradi and maxgradl point to the next \tau-violating variable to be considered. If there is no such variable, optimization is finished, so break.
		if(maxgradi==-1 && maxgradl==-1)
		{
			check_again();
			if(maxgradi==-1 && maxgradl==-1)
			{
		  
		  		break;
			}
		}
		//otherwise
		itns++;
		totitns++;
		
		//try and do a double step
		dualprogress=double_step();
		dual+=dualprogress;
		
		if(totitns%1000==0)
		{
			
			printf(".");
			fflush(stdout);
			
		}
		
		
		//if it's time to shiftt to a new label, update all the other gradients
		if( !check_shift(itns, dualprogress) || full)
		{
		  	
		  	itns=0;
		  	update_all_gradients(maxgradl);
			
		}
	}while(check_whole(totitns, dualprogress));

	double end_time=getTime();
	
	std::cout<<"\nTime ="<<end_time-start_time<<"\n";
 	std::cout<<"Iterations ="<<totitns<<"\n";
	
}

M3LFloat M3LKernel::double_step()
{
	//if maxgradi=-1, all gradients along this label have nmagnitude less than tau, so do nothing and return
    	if(maxgradi==-1) return 0;
	
	//first try to find another point along this label which has non zero PG, and that maximizes second order progress
    	M3LFloat g1=gradients[maxgradl][maxgradi];
    	int p=maxgradi;
    	int l=maxgradl;
    	M3LFloat* K_p=get_K(p);
    	M3LFloat score ;
    	M3LFloat g1sq=g1*g1;
    	M3LFloat* grads=gradients[maxgradl];
    	M3LFloat* theta=Theta[maxgradl];
    	M3LFloat tmp,g2sq;
	M3LFloat maxscore=0;
   	int q=-1;
   	M3LFloat denom=0;
    	for(int i=0;i<N;i++)
    	{
	
		denom=(K_ii[i]*K_ii[p] - K_p[i]*K_p[i]);
	
		g2sq=grads[i]*grads[i];
		score =(g1sq * K_ii[i] + g2sq * K_ii[p] - 2*grads[i]*g1*K_p[i])/denom;
		if(i!=p && PGs[i] && grads[i]*g1<0 && score>maxscore && denom!=0)
	  	{
	    		maxscore=score;
	    		q=i;
	  	}
	
    	}
    
	//if you cannot find such a point do a single step along p
    	if(q==-1  )
    	{
      
      		return single_step(maxgradi, maxgradl);
    	}
    	else
    	{
		//double step
      		SparseVector &xp=prob->data[p].inpt;
		SparseVector &xq=prob->data[q].inpt;
		M3LFloat Kpp=K_ii[p];
		M3LFloat Kqq=K_ii[q];
		M3LFloat Kpq=K_p[q];
		M3LFloat g2=grads[q];
		denom=(K_ii[q]*K_ii[p] - K_p[q]*K_p[q])*R.get(l,l);
		M3LFloat lbp=0-Theta[l][p];
		M3LFloat lbq=0-Theta[l][q];
		M3LFloat ubp=C2-Theta[l][p];
		M3LFloat ubq=C2-Theta[l][q];
		
		//Solving a 2D quadratic program with box constraints. Not analytic, but a sequence of if then does the job. 	
		M3LFloat deltap=(Kqq*g1-Kpq*g2)/denom;
		M3LFloat deltaq=(Kpp*g2-Kpq*g1)/denom;
		bool pbound=deltap>=lbp && deltap<=ubp;
		bool qbound=deltaq>=lbq && deltaq<=ubq;

		//if no variable is at its bound, update and return
		if(pbound && qbound)
		{
			//update and return
		  	update_double_label(p,q,l, deltap, deltaq, K_p);
			
		  	return -0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		}

		//else look at edges
		if(!pbound)
		{
		   	deltap=mymin(ubp,deltap);
		    	deltap=mymax(lbp,deltap);
		  	deltaq=(g2-Kpq*deltap)/Kqq;
		  	if(deltaq>=lbq && deltaq<=ubq)
		  	{
				//update and return
		  		update_double_label(p,q,l, deltap, deltaq, K_p);
				
		  		return -0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		  	}
		}
		if(!qbound)
		{
		  	deltaq=mymin(ubq,deltaq);
		  	deltaq=mymax(lbq, deltaq);
		  	deltap=(g1-Kpq*deltaq)/Kpp;
		  	if(deltap>=lbp && deltap<=ubp)
		  	{
				//update and return
		  		update_double_label(p,q,l, deltap, deltaq, K_p);
				
		  		return -0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		  	}
		}

		//finally look at corners
		M3LFloat deltap1=lbp;
		M3LFloat deltaq1=lbq;
		deltap=lbp;
		deltaq=lbq;
		M3LFloat dp1=-0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		deltap=ubp;
		M3LFloat dp=-0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		if(dp>dp1)
		{
		  	deltap1=deltap;
		  	deltaq1=deltaq;
		  	dp1=dp;
		}
		deltaq=ubq;
		dp=-0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		if(dp>dp1)
		{
		  	deltap1=deltap;
		  	deltaq1=deltaq;
		  	dp1=dp;
		}
		deltap=lbp;
		dp=-0.5*(Kpp*deltap*deltap+Kqq*deltaq*deltaq+2*Kpq*deltap*deltaq)*R.get(l, l)+g1*deltap+g2*deltaq;
		if(dp>dp1)
		{
		  	deltap1=deltap;
		  	deltaq1=deltaq;
		  	dp1=dp;
		}
		
		//update and return
		update_double_label(p,q,l, deltap1, deltaq1, K_p);
		
		return dp1;
    	}


}

M3LFloat M3LKernel::single_step(int p, int label)
{
	//solving a 1D QP with box constraints : -1/2 ax^2+bx
	M3LFloat b=gradients[label][p];
	if(fabs(b)<tau)
	{
	 	return 0;
	}
	SparseVector &xp=prob->data[p].inpt;
	M3LFloat a=K_ii[p]*R.get(label, label);
	M3LFloat lb=0-Theta[label][p];
	M3LFloat ub=C2-Theta[label][p];
	M3LFloat delta=b/a;
	delta=mymin(ub, mymax(lb, delta));
	update_label(p, label, delta);
	return -0.5*delta*delta*a+b*delta;
	
}
void M3LKernel::update_label(int p, int label, M3LFloat delta)
{
	//single step update
	//update all gradients along this label and store up information needed for updating other labels later
	Theta[label][p]+=delta;
	M3LFloat* Kp=get_K(p);
	M3LFloat Rll=R.get(label, label);
	M3LFloat dg;
	M3LFloat* grads=gradients[label];
	M3LFloat* theta=Theta[label];
	maxgradi=-1;
	M3LFloat tmp;
	M3LFloat gmax=tau;
	M3LFloat gmax2=tau;
	maxupdate=0;
	for(int i=0;i<N;i++)
	{
		dg=Kp[i]*delta;
		grads[i]-=dg*Rll;
		tmp=(theta[i]>=C2 && grads[i]>=0)?0:grads[i];
		tmp=(theta[i]<=0 && grads[i]<=0)?0:tmp;
		tmp=fabs(tmp);
		PGs[i]=tmp;
		if(tmp>gmax && tmp>tau)
		{
		  	maxgradi=i;
		
			
		  	gmax=tmp;
		  	
		}
		//store up updates in current update so we can update other labels later
		current_update[i]+=dg;
		maxupdate=mymax(maxupdate, fabs(current_update[i]));
	}
	maxupdate=maxR[maxgradl]*maxupdate;
}
void M3LKernel::update_double_label(int p, int q, int label, M3LFloat deltap, M3LFloat deltaq, M3LFloat* Kp)
{
	//double step update
	//update all gradients along this label and store up information needed for updating other labels later
	Theta[label][p]+=deltap;
	Theta[label][q]+=deltaq;
	
	M3LFloat* Kq=get_K(q);
	M3LFloat Rll=R.get(label, label);
	M3LFloat dg;
	M3LFloat* grads=gradients[label];
	M3LFloat* theta=Theta[label];
	M3LFloat gmax=tau,tmp;
	maxgradi=-1;
	maxupdate=0;
	M3LFloat gmax2=tau;
	for(int i=0;i<N;i++)
	{
		dg=Kp[i]*deltap + Kq[i]*deltaq;
		grads[i]-=dg*Rll;
		tmp=((theta[i]>=C2 && grads[i]>=0)||(theta[i]<=0 && grads[i]<=0))?0:grads[i];
		tmp=fabs(tmp);
		PGs[i]=tmp;
		
		if(tmp>gmax)
		{
		  	maxgradi=i;
			gmax=tmp;
		  
		}
		
		//store up updates in current update so we can update other labels later
		current_update[i]+=dg;
		maxupdate=mymax(maxupdate, fabs(current_update[i]));
	}
	maxupdate=maxR[maxgradl]*maxupdate;
}

void M3LKernel::update_all_gradients(int label)
{
	//update all gradients
	const M3LFloat* Rl=R.getRow(label);
	M3LFloat gmax,tmp;
	if(maxgradi!=-1)
	{
	
		gmax=fabs(gradients[maxgradl][maxgradi]);
	}
	else
	{
		gmax=0;
		maxgradi=-1;
		maxgradl=-1;
	}
	M3LFloat gmax2=tau;
	for(int i=0;i<L;i++)
	{
		if(i==label) 
			continue;
		M3LFloat* grads=gradients[i];
		M3LFloat* theta=Theta[i];
		for(int j=0;j<N;j++)
		{

			M3LFloat prev=grads[j];
			grads[j]-=Rl[i]*current_update[j];
			tmp=(theta[j]>=C2 && grads[j]>=0)?0:grads[j];
			tmp=(theta[j]<=0 && grads[j]<=0)?0:tmp;
			tmp=fabs(tmp);
			if(tmp>gmax && tmp>tau)
			{
			  	maxgradi=j;
			  	maxgradl=i;
				gmax2=gmax;
			  	gmax=tmp;
		  
			}
			else if(tmp>gmax2)
			{
				gmax2=tmp;
			}

		}
		
	}
	maxgrad2=gmax2;
	maxupdate=0;
	for(int i=0;i<N;i++)
	{
		current_update[i]=0;
	}
	if(maxgradl==-1) return;
	int j=maxgradl;
	for(int i=0;i<N;i++)
	{
		tmp=(Theta[j][i]>=C2 && gradients[j][i]>=0)?0:gradients[j][i];
		tmp=(Theta[j][i]<=0 && gradients[j][i]<=0)?0:tmp;
		PGs[i]=fabs(tmp);
	}
	
}





M3LModel* M3LKernel::learn(M3LProblem* problem, M3LParameters* params)
{
	train();
	M3LModel* model=new M3LModel;
	model->L=L;
	model->kparams=kparams;
	model->AlgoType=ALG_KER;
	model->bias=bias;
	
	model->Zt=FullMatrix::zero(L,d);
	model->SVs.reserve(N);
	M3LFloat th;
	bool flag;
	FullMatrix M;
	
	for(int i=0;i<N;i++)
	{
		FullVector theta;
		theta.create(L,0);
		flag=false;
		for(int j=0;j<L;j++)
		{
			th=train_labels[i][j]>0?Theta[j][i]:Theta[j][i]-C2;
			if(fabs(th)>1e-9)
			{
				theta.set(j,th);
				flag=true;
			}
		}
		
		theta=mult(R,theta);
		
		if(flag)
		{
			model->SVs.push_back(M3LModel::SupportVector());
			
			model->SVs[model->SVs.size()-1].theta = theta;
			model->SVs[model->SVs.size()-1].x =  prob->data[i].inpt;
			
			
		}
	}
	
	model->R=R;
	
	return model;
}
void M3LKernel::destroy()
{
	for(int i=0;i<L;i++)
	{
		delete[] gradients[i];
		delete[] Theta[i];
	}
	delete[] gradients;
	delete[] Theta;
	delete[] train_labels;
	delete[] current_update;
	delete[] K_ii;
	delete[] PGs;
	delete cache;
}
void M3LKernel::check_again()
{
	M3LFloat gmax=0,tmp;
	maxgradi=-1;
	maxgradl=-1;
	for(int i=0;i<L;i++)
	{
		
		M3LFloat* grads=gradients[i];
		M3LFloat* theta=Theta[i];
		for(int j=0;j<N;j++)
		{

			tmp=(theta[j]>=C2 && grads[j]>=0)?0:grads[j];
			tmp=(theta[j]<=0 && grads[j]<=0)?0:tmp;
			tmp=fabs(tmp);
			if(tmp>gmax && tmp>tau)
			{
			  	maxgradi=j;
			  	maxgradl=i;
			  	gmax=tmp;
		  
			}

		}
		
	}
	
}
