#include <fftw3.h>
#include <armadillo>
#include "fftw1DR.h"

using namespace std;
using namespace arma;

CFFTW1DR::CFFTW1DR() 
{
  bool newflag = true;
  for (int i = 0; i < MAXFFTEXP2NUM; i++) {
    int n = 1 << i;
    m_in_buf[i] = (f8*) fftw_malloc(sizeof(f8) * n);
    m_out_cbuf[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    m_plan[i] = fftw_plan_dft_r2c_1d(n, m_in_buf[i], m_out_cbuf[i], FFTW_MEASURE);
    newflag = (m_in_buf[i] && m_out_cbuf[i] && m_plan[i]);
  }
  if (!newflag) {
    cout << "Error: memory new failed." << endl;
    release();
  }
}

CFFTW1DR::~CFFTW1DR() 
{
  release();
}

i4 CFFTW1DR::release()
{
  for (int i=0; i<MAXFFTEXP2NUM; i++) {
    if (m_in_buf[i]) {
      fftw_free(m_in_buf[i]);
      m_in_buf[i] = NULL;
    }
    if (m_out_cbuf[i]) {
      fftw_free(m_out_cbuf[i]);
      m_out_cbuf[i] = NULL;
    }
    if (m_plan[i]) {
      fftw_destroy_plan(m_plan[i]);
    }
  }
  return 0;
}

int CFFTW1DR::fftw_real_1d_with_half_ret(const mat& A, cx_mat& B)
{
  i4 n=A.n_elem;
  i4 no = n/2 + 1;
  if (B.n_elem < no) B.resize(no, 1);   // first no element matters for 1D real fft
  f8 *in=NULL;
  fftw_complex *out=NULL;
  fftw_plan p;

  i4 exp_idx = is_exp2(n);
  if (exp_idx >= 0) {
    // power of 2
    in = m_in_buf[exp_idx];
    out = m_out_cbuf[exp_idx];
    p = m_plan[exp_idx];
  } else {
    // other length
    in = (f8*) fftw_malloc(sizeof(f8) * n);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * no);
    p = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
    if (!in || !out) return -1;
  }

  memcpy(in, A.memptr(), n*sizeof(f8));
  fftw_execute(p); // ִ�б任
  memcpy(B.memptr(), out, no*sizeof(f8)*2);

  return 0;
}

int CFFTW1DR::is_exp2(int num)
{
  switch (num)
  {
  case 1:return 0; break;
  case 2:return 1; break;
  case 4:return 2; break;
  case 8:return 3; break;
  case 16:return 4; break;
  case 32:return 5; break;
  case 64:return 6; break;
  case 128:return 7; break;
  case 256:return 8; break;
  case 512:return 9; break;
  case 1024:return 10; break;
  case 2048:return 11; break;
  case 4096:return 12; break;
  case 8192:return 13; break;
  case 16384:return 14; break;
  case 32768:return 15; break;
  default: return -1; break;
  }
  return -1;
}

