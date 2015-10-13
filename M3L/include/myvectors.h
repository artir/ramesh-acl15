#ifndef MYVECTORS_H
#define MYVECTORS_H 1
#include "definitions.h"
#include <ios>
#include <iomanip>
#include <iostream>
#include <cassert>

//A small vector library

//forward declarations
class FullMatrix;
class FullVector;

struct svm_node
{
	int index;
	M3LFloat value;
	svm_node(){}
	svm_node(int index, M3LFloat value):index(index), value(value){}
	
};

//sparse vector
class SparseVector
{
	svm_node* p;
	
	int num_nonzero;
	public:
	
	SparseVector(){ p=0; num_nonzero=0;}
	
	SparseVector(const SparseVector& s):num_nonzero(s.num_nonzero)
	{
		
		p=new svm_node[num_nonzero];
		for(int i=0;i<num_nonzero;i++)
		{
			p[i].index=s.p[i].index;
			p[i].value=s.p[i].value;
		}
		
	}
	~SparseVector()
	{
		if(p!=0) delete[] p;
	}
	SparseVector& operator=(const SparseVector& rhs)
	{
		if(p!=0) delete[] p;
		num_nonzero=rhs.num_nonzero;
		p=new svm_node[num_nonzero];
		for(int i=0;i<num_nonzero;i++)
		{
			p[i].index=rhs.p[i].index;
			p[i].value=rhs.p[i].value;
		}
		
		return *this;
	}
	void clear()
	{
		if(p!=0)
		{
			delete[] p;
			p=0;
			num_nonzero=0;
		}
		
	}
	void create(int n)
	{
		p=new svm_node[n];
		for(int i=0;i<n;i++)
		{
			p[i].index=0;
			p[i].value=0.0;
		}
		num_nonzero=0;
		
	}
	
	int last(){ return num_nonzero>0?p[num_nonzero-2].index:0;}
	int get_nnz(){ return num_nonzero;}
	int get_ith_index(int i)
	{
		if(p==0) return 0;
		return p[i].index;
	}
	M3LFloat get_ith_value(int i)
	{
		if(p==0) return 0;
		return p[i].value;
	}
	void print_libsvm(std::ostream& f);
	void read_libsvm(std::istream& f);
	friend FullVector mult(const FullMatrix& M, const SparseVector& x);
	friend M3LFloat dot(const SparseVector& px, const SparseVector& py);
	friend std::ostream& operator<<(std::ostream& fout, const SparseVector& x);
	friend std::istream& operator>>(std::istream& fin, SparseVector& x);
	friend FullMatrix outerprod(FullVector v, SparseVector x);
		
};

class FullVector
{
	M3LFloat* data;
	
	int d;
	public:
	FullVector()
	{
		data=0;
		d=0;
	}
	FullVector(int sz, M3LFloat id=0)
	{
		d=sz;
		data=new M3LFloat[d];
		for(int i=0;i<d;i++)
		{
			data[i]=id;
		}
	}
	FullVector(const FullVector& vec)
	{

		
		d=vec.d;
		data=new M3LFloat[d];
		for(int i=0;i<d;i++)
		{
			data[i]=vec.data[i];
		}
		
	}
	~FullVector()
	{
		delete[] data;
	}
	void clear()
	{
		if(data!=0)
		{
			delete[] data;
			data=0;
		}
		d=0;
	}
	void create(int n, M3LFloat id=0)
	{
		d=n;
		data=new M3LFloat[n];
		for(int i=0;i<d;i++)
			data[i]=id;
	}
	M3LFloat get(int i)
	{
		return data[i];
	}
	void set(int i, M3LFloat v)
	{
		data[i]=v;
	}
	int size()
	{
		return d;
	}
	M3LFloat* getData()
	{
		return data;
	}
	FullVector& operator=(const FullVector& vec)
	{
		if(data!=0) delete[] data;
		d=vec.d;
		data=new M3LFloat[d];
		for(int i=0;i<d;i++)
		{
			data[i]=vec.data[i];
		}
		return *this;
	}
	void add(const FullVector& other, const M3LFloat& scale=1.f);
	friend FullVector mult(const FullMatrix& M, const FullVector& x);
	friend M3LFloat dot(const FullVector& px, const FullVector& py);
	friend std::ostream& operator<<(std::ostream& fout, const FullVector& x);
	friend std::istream& operator>>(std::istream& fin, FullVector& x);
	friend FullMatrix outerprod(FullVector v, SparseVector x);
	friend FullVector mult(const FullMatrix& M, const SparseVector& x);
};
class FullMatrix
{
	public:
	int m;
	int n;
	M3LFloat* data;
	FullMatrix()
	{
		data=0;
		m=0;
		n=0;
	}
	FullMatrix(int m, int n):m(m), n(n)
	{
		data=new M3LFloat[m*n];
	}
	FullMatrix(int m, int n, M3LFloat id):m(m), n(n)
	{
		data=new M3LFloat[m*n];
		for(int i=0;i<m*n;i++)
			data[i]=id;
	}
	FullMatrix(const FullMatrix& M):m(M.m),n(M.n) 
	{
		data=new M3LFloat[m*n];
		for(int i=0;i<m*n;i++)
			data[i]=M.data[i];
	}
	~FullMatrix()
	{
		delete[] data;
	}
	static FullMatrix identity(int n1)
	{
		FullMatrix M;
		M.create(n1,n1);
		for(int i=0;i<n1*n1;i++)
		{
			M.data[i]=0.f;
		}
		for(int i=0;i<n1;i++)
		{
			M.data[i*n1+i]=1.f;
		}
		return M;
	}
	static FullMatrix zero(int m1, int n1)
	{
		FullMatrix M;
		M.create(m1,n1);
		for(int i=0;i<m1*n1;i++)
		{
			M.data[i]=0;
		}
		
		return M;
	}
	FullMatrix& operator=(const FullMatrix& M)
	{
		if(data!=0) delete[] data;
		m=M.m;
		n=M.n;
		data=new M3LFloat[m*n];
		for(int i=0;i<m*n;i++)
			data[i]=M.data[i];
		return *this;
	}
	void clear()
	{
		if(data!=0)
		{
			delete[] data;
			data=0;
		}
		m=0;
		n=0;
	}
	void create(int m1, int n1)
	{
		m=m1;
		n=n1;
		data=new M3LFloat[m*n];
		for(int i=0;i<m*n;i++)
			data[i]=0;
	}
	M3LFloat get(int i, int j)
	{
		return data[i*n+j];
	}
	void set(int i, int j, M3LFloat v)
	{
		data[i*n+j]=v;
	}
	void add(const FullMatrix& other, const M3LFloat& scale=1.f);
	M3LFloat* getRow(int i)
	{
		return &data[i*n];
	}
	friend FullVector mult(const FullMatrix& M, const FullVector& x);
	friend FullVector mult(const FullMatrix& M, const SparseVector& x);
	friend std::ostream& operator<<(std::ostream& fout, const FullMatrix& x);
	friend std::istream& operator>>(std::istream& fin, FullMatrix& x);
	friend FullMatrix outerprod(FullVector v, SparseVector x);

};








FullVector mult(const FullMatrix& M, const FullVector& x);
FullVector mult(const FullMatrix& M, const SparseVector& x);
inline M3LFloat dot(const SparseVector& px, const SparseVector& py)
{
	M3LFloat sum = 0;
	svm_node* p1=px.p;
	svm_node* p2=py.p;
	if(p1==0 || p2==0) return sum;
	while(p1->index != -1 && p2->index != -1)
	{
		if(p1->index == p2->index)
		{
			sum += p1->value * p2->value;
			++(p1);
			++(p2);
		}
		else
		{
			if(p1->index > p2->index)
				++p2;
			else
				++p1;
		}			
	}
	
	return sum;
}

M3LFloat dot(const FullVector& px, const FullVector& py);
std::ostream& operator<<(std::ostream& fout, const SparseVector& x);
std::istream& operator>>(std::istream& fin, SparseVector& x);
std::ostream& operator<<(std::ostream& fout, const FullVector& x);
std::istream& operator>>(std::istream& fin, FullVector& x);
std::ostream& operator<<(std::ostream& fout, const FullMatrix& x);
std::istream& operator>>(std::istream& fin, FullMatrix& x);
FullMatrix outerprod(FullVector v, SparseVector x);
#endif
