#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <sys/time.h> 
#include <vector>

#include "../CombBLAS.h"

using namespace std;

static double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define CUBE(a) (a * a * a)

//= Define our custom reducers =//
MPI_Op MPI_SUMSQUARES;
MPI_Op MPI_SUMCUBES;

template <typename T>
void sumSquares(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
{
	T *in = static_cast<T *>(invec);
	T *out = static_cast<T *>(inoutvec);

	for (int i=0; i<*len; i++)
		out[i] = in[i]*in[i] + out[i]*out[i];
}

template<typename T>
struct sum_squares : public std::binary_function<T, T, T>
{
	T operator()(const T &x, const T &y) const
	{
		return x*x + y*y;
	}
};

template<typename T> struct MPIOp<sum_squares<T>, T>
{
	static MPI_Op op()
	{
		MPI_Op_create(sumSquares<double>, false, &MPI_SUMSQUARES);
		return MPI_SUMSQUARES;
	}
};

//= Calculate the distance between two points =//
//= Adapted from Numerical Recipes in C++ Ch. 2.6 =//
double
pythag
(double a, double b)
{
	float absa, absb;

	absa = fabs(a);
	absb = fabs(b);

	if (absa > absb)
		return absa * sqrt(1.0 + SQR(absb / absa));
	else
		return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

#ifdef _OPENMP
int cblas_splits = omp_VecMax_threads(); 
#else
int cblas_splits = 1;
#endif

typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols<int64_t, NT> DCCols;
	typedef SpParMat<int64_t, NT, DCCols> MPI_DCCols;
};

//= Distributed Eigenvalue solver for symmetrical tridiagonal matrices =//
//= d -- vector representing the diagonal of the input matrix =//
//= e -- vector representing the off-diagonals of the input matrix =//
//= n -- number of elements in d
//= Note: e has size n and contains n-1 elements. It is expected that the =//
//= first element is empty =//
//= Adapted from Numerical Recipes in C++ Ch. 11.3 =//
FullyDistVec<int64_t, double>
Eigen
(FullyDistVec<int64_t, double> d, FullyDistVec<int64_t, double> e, int n)
{
	int m, l, iter, i, k;
	double s, r, p, g, f, dd, c, b;
	
	e.SetElement(n-1, 0.0);

	for (l=0; l<n; l++) {
		iter = 0;
		do {
			for (m=l; m<n-1; m++) {
				dd = fabs(d[m]) + fabs(d[m+1]);
				if (fabs(e[m]) + dd == dd)
					break;
			}

			if (m != l) {
				if (iter++ == 30) {
					cout << "Too many iterations" << endl;
				}
				g = (d[l+1] - d[l]) / (2.0 * e[l]);
				r = pythag(g, 1.0);
				g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
				s = c = 1.0;
				p = 0.0;
				for (i=m-1; i>=l; i--) {
					f = s * e[i];
					b = c * e[i];
					r = pythag(f, g);
					e.SetElement(i+1, r);
					if (r == 0.0) {
						d.SetElement(i+1, d[i+1] - p);
						e.SetElement(m , 0.0);
						break;
					}
					s = f / r;
					c = g / r;
					g = d[i+1] - p;
					r = (d[i] - g) * s + 2.0 * c * b;
					p = s * r;
					d.SetElement(i+1, g + p);
					g = c * r - b;
				}
				if (r == 0.0 && i>= l)
					continue;
				d.SetElement(l,d[l]-p);
				e.SetElement(l , g);
				e.SetElement(m, 0.0);
			}
		} while (m != l);
	}
	return d;
}

//= Calculate the sum of the elements in a vector =//
//= FIXME CombBLAS should be able to do this, no? =//
double
VecSumCubes
(FullyDistVec<int64_t, double> vec, int l)
{
	double sum = 0;

	sum = vec.Reduce(plus<double>(), (double) 0);
	for (int i=0; i<l; i++)
		sum += vec[i] * vec[i] * vec[i];

	return sum;
}

//= Calculate the sum of the elements in a vector =//
//= FIXME CombBLAS should be able to do this, no? =//
double
VecAbsSumCubes
(FullyDistVec<int64_t, double> vec, int l)
{
	double sum = 0;

	for (int i=0; i<l; i++)
		sum += fabs(vec[i] * vec[i] * vec[i]);

	return sum;
}

//= Calculate the norm of the elements in a vector =//
//= FIXME CombBLAS should be able to do this, no? =//
double
VecNorm
(FullyDistVec<int64_t, double> vec, int size)
{
	double sum;

	cout << "Reducer: " << vec.Reduce(sum_squares<double>(), 0.0) << endl;

	for (int i=0; i<size; i++)
			sum += vec[i] * vec[i];

	cout << "Loop: " << sum << endl;

	return sqrt(sum);
}

// vector dot product
//= Calculate a the dot product of two vectors =//
double
VecDotProduct
(FullyDistVec<int64_t, double> v1, FullyDistVec<int64_t, double> v2, int size)
{
	double result;

	FullyDistVec<int64_t, double> tmp = v1;
	tmp.EWiseApply(v2, std::multiplies<double>());
	result = tmp.Reduce(std::plus<double>(), 1);

	return result;
}

//= Re-orthoganalization for Lanczos algorithm =//
//= performs Q*(Q'*z) =//
FullyDistVec<int64_t, double>
Reorthog
(FullyDistVec<int64_t, double> z, FullyDistVec<int64_t, double> *Q, int i, int n)
{
	FullyDistVec<int64_t, double> tmp_ix1(i+1, 0);
	FullyDistVec<int64_t, double> tmp_nx1(n, 0);
	FullyDistVec<int64_t, double> QRowI_ix1(i+1, 0);

	for(int j=0; j<=i; j++)
		tmp_ix1.SetElement(j, VecDotProduct(Q[j], z, n));

	for(int j=0; j<n; j++) {	
		for(int k=0; k<=i; k++)
			QRowI_ix1.SetElement(k, Q[k][j]);
		tmp_nx1.SetElement(j, VecDotProduct(QRowI_ix1, tmp_ix1, i+1));
	}

	return tmp_nx1;
}

//= Lanczos method for approximating Eigen values =//
//= This implementation reuses alpha, beta and Q to avoid redundancy =//
//= Adapted from 
FullyDistVec<int64_t, double>
Lanczos
(PSpMat<double>::MPI_DCCols A, FullyDistVec<int64_t, double> &alpha,
 FullyDistVec<int64_t, double> &beta, FullyDistVec<int64_t, double> *Q, int m)
{
		int n = A.getnrow();
		FullyDistVec<int64_t, double> qq;
		FullyDistVec<int64_t, double> z;
		FullyDistVec<int64_t, double> eigenvalues;

		qq = Q[m-1];

		z = SpMV<PTDOUBLEDOUBLE>(A, qq);
		alpha.SetElement(m-1, VecDotProduct(qq, z, n));

		z -= Reorthog(z, Q, m-1, n);
		z -= Reorthog(z, Q, m-1, n);
		beta.SetElement(m-1, VecNorm(z, n));

		for (int k=0; k<n; k++)
			qq.SetElement(k, z[k] / beta[m-1]);

		Q[m] = qq;

		eigenvalues = Eigen(alpha, beta, m);

		//= Dump Eigenvalues =//
		//for(int i=0; i<m; i++)
			//cout << eigenvalues[i] << endl;

		return eigenvalues;
}

//= Find largest element in vector =//
double
VecMax
(FullyDistVec<int64_t, double> lanczosResult, int size)
{
	double max = lanczosResult[0];

	for (int i=1; i<size; i++)
		if (lanczosResult[i] > max)
			max = lanczosResult[i];

	return max;
}

//= Find smallest element in vector =//
double
VecMin
(FullyDistVec<int64_t, double> lanczosResult, int size)
{
	double min = fabs(lanczosResult[0]);

	for (int i=1; i<size; i++)
		if (fabs(lanczosResult[i]) < min)
			min = lanczosResult[i];

	return min;
}

//= Driver for EigenTriangle algorithm =//
int
main
(int argc, char* argv[])
{
	//= MPI setup =//
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	//= Validate the arguments =//
	if(argc < 2) {
		if(myrank == 0)
			cout << "Usage: ./MultTest <graph_data>" << endl;
		MPI_Finalize(); 
		return -1;
	}				

	//= EigenTriangle Algorithm setup =//
	double tolerance = .000001;

	//= A is the adjacency matrix =//
	PSpMat<double>::MPI_DCCols A;
	string GraphData = string(argv[1]);
	A.ReadDistribute(GraphData, 0);
	int n = A.getnrow();

	//= Lanczos Algorithm setup =//
	const int mMax = 1000;
	FullyDistVec<int64_t, double> alpha(mMax, 0);
	FullyDistVec<int64_t, double> beta(mMax, 0);
	FullyDistVec<int64_t, double> *Q = new FullyDistVec<int64_t, double>[mMax+1];
	Q[0]= FullyDistVec<int64_t, double>(n,1);
	Q[0].SetElement(n-2,0.00001); //= Why? I don't know! =//
	double norm = VecNorm(Q[0], n);
	for (int i=0; i<n; i++) {
		Q[0].SetElement(i, Q[0][i] / norm);
	}

	int i = 1;				//= Iteration =//
	FullyDistVec<int64_t, double> lambda(mMax, 0);
	double minLambda;
	double lambda3 = 0;		//= Cube of Eigenvalue =//
	double sumLambda3 = 0;	//= Sum of cubes of Eigenvalues =//
	do {
		lambda = Lanczos(A, alpha, beta, Q, i);
		MPI_Barrier(MPI_COMM_WORLD);
		minLambda = VecMin(lambda, i);
		lambda3 = CUBE(minLambda);
		sumLambda3 = VecSumCubes(lambda, i);

		if (myrank == 0) {
			cout << "lambda[" << i << "] = " << minLambda << endl;
			cout << "triangle count: " << sumLambda3 / 6 << endl;
			cout << "convergence: " << lambda3 / sumLambda3 << " > " << tolerance << endl;
			cout << "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~" << endl;
		}

		i++;
	//} while (fabs(lambda3) / sumLambda3 > tolerance || lambda3 / sumLambda3 < 0);
	} while ((lambda3 / sumLambda3 > tolerance) || (lambda3 / sumLambda3 < 0));

	if(myrank==0)
		cout << "Triangle count: " << sumLambda3 / 6 << " after " << i << " iterations." << endl;

	MPI_Finalize();
	return 0;
}
