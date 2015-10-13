#include "myvectors.h"

void FullVector::add(const FullVector& other, const M3LFloat& scale)
{
	assert(d==other.d);
	for(int i=0;i<d;i++)
	{
		data[i]+=other.data[i]*scale;
	}
}
void SparseVector::print_libsvm(std::ostream& fout)
{
	svm_node* p1=p;
	if(p1==0) return;
	std::streamsize oldprec = fout.precision();
  	fout << std::scientific << std::setprecision(sizeof(M3LFloat)==4 ? 7 : 16);
  
	while(p1->index != -1)
	{
		fout<<" "<<p1->index+1<<":"<<p1->value;
		p1++;
	}
	fout<<"\n";
	fout<< std::scientific << std::setprecision(oldprec);
	
}
void SparseVector::read_libsvm(std::istream& f)
{
	
	std::vector<svm_node> nodes;
	int i;
	char c;
	M3LFloat x;
	for(;;)
    	{
      		c = f.get();
      		if (!f.good() || c=='\n' )
        		break;
     		if (::isspace(c))
        		continue;
      		
      		f.unget();
      		f>>i;
		i--;
      		if (f.get() != ':')
        	{
          		f.unget();
          		break;
        	}
      		
      		f>>x;
		if (!f.good())
        		break;
      		nodes.push_back(svm_node(i,x));
    	}
	
   	clear();
	
    	create(nodes.size()+1);
	
	for(int i=0;i<nodes.size();i++)
	{
		p[i]=nodes[i];
	}
	num_nonzero=nodes.size()+1;
	p[nodes.size()].index=-1;
	
}
void FullMatrix::add(const FullMatrix& other, const M3LFloat& scale)
{
	assert(m==other.m);
	assert(n==other.n);
	for(int i=0;i<m*n;i++)
	{
		data[i]+=other.data[i]*scale;
	}
}



M3LFloat dot(const FullVector& px, const FullVector& py)
{
	assert(px.d==py.d);
	M3LFloat sum=0;
	for(int i=0;i<px.d;i++)
	{
		sum+=px.data[i]*py.data[i];
	}
	return sum;
}


std::ostream& operator<<(std::ostream& fout, const SparseVector& x)
{
	svm_node* p=x.p;
	if(p==0) return fout;
	std::streamsize oldprec = fout.precision();
  	fout << std::scientific << std::setprecision(sizeof(M3LFloat)==4 ? 7 : 16);
  
	while(p->index != -1)
	{
		fout<<" "<<p->index<<":"<<p->value;
		p++;
	}
	fout<< std::scientific << std::setprecision(oldprec);
	return fout;
}

std::ostream& operator<<(std::ostream& fout, const FullVector& x)
{
	
	std::streamsize oldprec = fout.precision();
  	fout << std::scientific << std::setprecision(sizeof(M3LFloat)==4 ? 7 : 16);
  
	for(int i=0;i<x.d;i++)
	{
		fout<<" "<<i<<":"<<x.data[i];
		
	}
	fout<<"\n";
	fout<< std::scientific << std::setprecision(oldprec);
	return fout;
}
std::ostream& operator<<(std::ostream& fout, const FullMatrix& M)
{
	
	std::streamsize oldprec = fout.precision();
  	fout << std::scientific << std::setprecision(sizeof(M3LFloat)==4 ? 7 : 16);
  
	for(int i=0;i<M.m;i++)
	{
		for(int j=0;j<M.n;j++)
		{
			fout<<" "<<i<<","<<j<<":"<<M.data[i*M.n+j];
		}
	}
	fout<< std::scientific << std::setprecision(oldprec);
	return fout;
}
std::istream& operator>>(std::istream& f, SparseVector& vec)
{
	std::vector<svm_node> nodes;
	int i;
	char c;
	M3LFloat x;
	for(;;)
    	{
      		c = f.get();
      		if (!f.good() || c=='\n')
        		break;
     		if (::isspace(c))
        		continue;
      		
      		f.unget();
      		f>>i;
      		if (f.get() != ':')
        	{
          		f.unget();
          		break;
        	}
      		
      		f>>x;
		if (!f.good())
        		break;
      		nodes.push_back(svm_node(i,x));
    	}
	
   	vec.clear();
	
    	vec.create(nodes.size()+1);
	
	for(int i=0;i<nodes.size();i++)
	{
		vec.p[i]=nodes[i];
	}
	
	vec.p[nodes.size()].index=-1;
	vec.num_nonzero=nodes.size();
  	return f;
}
std::istream& operator>>(std::istream& f, FullVector& vec)
{
	std::vector<svm_node> nodes;
	int i;
	char c;
	M3LFloat x;
	int maxindex=0;
	for(;;)
    	{
      		c = f.get();
      		if (!f.good() || c=='\n')
        		break;
     		if (::isspace(c))
        		continue;
      		
      		f.unget();
      		f>>i;
		
      		if (f.get() != ':')
        	{
          		f.unget();
          		break;
        	}
      		
      		f>>x;
		if (!f.good())
        		break;
		maxindex=mymax(i, maxindex);
      		nodes.push_back(svm_node(i,x));
    	}
   	vec.clear();
	vec.create(maxindex+1);
	for(int i=0;i<nodes.size();i++)
	{
		vec.data[nodes[i].index]=nodes[i].value;
	}

  	return f;
}

std::istream& operator>>(std::istream& f, FullMatrix& M)
{
	std::vector<int> index1;
	std::vector<int> index2;
	std::vector<M3LFloat> value;
	int i,j;
	char c;
	M3LFloat x;
	int maxindexi=0;
	int maxindexj=0;
	for(;;)
    	{
      		c = f.get();
      		if (!f.good() || c=='\n')
        		break;
     		if (::isspace(c))
        		continue;
      		
      		f.unget();
      		f>>i;
		if(f.get()!=',')
		{
			f.unget();
          		f.setstate(std::ios::badbit);
          		break;
		}
		f>>j;
      		if (f.get() != ':')
        	{
          		f.unget();
          		f.setstate(std::ios::badbit);
          		break;
        	}
      		
      		f>>x;
		if (!f.good())
        		break;
		maxindexi=mymax(i, maxindexi);
		maxindexj=mymax(j, maxindexj);
      		index1.push_back(i);
		index2.push_back(j);
		value.push_back(x);
    	}
   	M.clear();
	M.create(maxindexi+1, maxindexj+1);
	for(int i=0;i<index1.size();i++)
	{
		M.set(index1[i], index2[i],value[i]);
	}

  	return f;
}
FullVector mult(const FullMatrix& M, const FullVector& vec)
{
	int d=M.m;
	assert(vec.d==M.n);
	FullVector result;
	result.create(d,0);
	M3LFloat* data=M.data;
	for(int i=0;i<d;i++)
	{
		for(int j=0;j<M.n;j++)
		{
			result.data[i]+=data[j]*vec.data[j];
		}
		data+=M.n;
	}
	return result;
}
FullVector mult(const FullMatrix& M, const SparseVector& vec)
{
	int d=M.m;
	
	FullVector result;
	result.create(d,0);
	M3LFloat* data=M.data;
	svm_node* p=vec.p;
	svm_node* q=p;
	if(q==0) return result;
	for(int i=0;i<d;i++)
	{

		
		while(q->index!=-1)
		{
			if(q->index>=M.n) break;
			result.data[i]+=data[q->index]*q->value;
			q++;
		}
		data+=M.n;
		q=p;
	}
	return result;
}
