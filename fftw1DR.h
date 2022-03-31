#ifndef  FFTW1DR_INC
#define  FFTW1DR_INC

#include <fftw3.h>
#include <armadillo>
#include "datatypes.h"

using namespace std;
using namespace arma;

#ifdef __cplusplus
extern "C" {
#endif


#define MAXFFTEXP2NUM 16
/*
 * =====================================================================================
 *        Class:  CFFTW1DR
 *  Description:  fast implementation of fft transformation with 1D real vector.
 * =====================================================================================
 */
class CFFTW1DR
{
public:
  /* ====================  LIFECYCLE     ======================================= */
  explicit CFFTW1DR ();                             /* constructor */
  ~CFFTW1DR();
  /* ====================  MUTATORS      ======================================= */
  int fftw_real_1d_with_half_ret(const mat& A, cx_mat& B);//Ò»Î¬¸´Êý
  /* ====================  OPERATORS     ======================================= */
 
protected:
	int is_exp2(int num);   // is exponential of 2

private:
  i4 release();
  /* ====================  DATA MEMBERS  ======================================= */
  fftw_plan m_plan[MAXFFTEXP2NUM];
  f8* m_in_buf[MAXFFTEXP2NUM];             // 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384
  fftw_complex* m_out_cbuf[MAXFFTEXP2NUM];
}; /* -----  end of class CFFTW1DR  ----- */


#ifdef __cplusplus
}
#endif

#endif   /* ----- #ifndef fftw1DR_INC  ----- */
