/* SLSQP: Sequentional Least Squares Programming (aka sequential quadratic programming SQP)
   method for nonlinearly constrained nonlinear optimization, by Dieter Kraft (1991).
   Fortran released under a free (BSD) license by ACM to the SciPy project and used there.
   C translation via f2c + hand-cleanup and incorporation into NLopt by S. G. Johnson (2009). */

/* Table of constant values */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "slsqp.h"

/*      ALGORITHM 733, COLLECTED ALGORITHMS FROM ACM. */
/*      TRANSACTIONS ON MATHEMATICAL SOFTWARE, */
/*      VOL. 20, NO. 3, SEPTEMBER, 1994, PP. 262-281. */
/*      http://doi.acm.org/10.1145/192115.192124 */


/*      http://permalink.gmane.org/gmane.comp.python.scientific.devel/6725 */
/*      ------ */
/*      From: Deborah Cotton <cotton@hq.acm.org> */
/*      Date: Fri, 14 Sep 2007 12:35:55 -0500 */
/*      Subject: RE: Algorithm License requested */
/*      To: Alan Isaac */

/*      Prof. Issac, */

/*      In that case, then because the author consents to [the ACM] releasing */
/*      the code currently archived at http://www.netlib.org/toms/733 under the */
/*      BSD license, the ACM hereby releases this code under the BSD license. */

/*      Regards, */

/*      Deborah Cotton, Copyright & Permissions */
/*      ACM Publications */
/*      2 Penn Plaza, Suite 701** */
/*      New York, NY 10121-0701 */
/*      permissions@acm.org */
/*      212.869.7440 ext. 652 */
/*      Fax. 212.869.0481 */
/*      ------ */

/********************************* BLAS1 routines *************************/

/*     COPIES A VECTOR, X, TO A VECTOR, Y, with the given increments */
static void dcopy___(int *n_, const double *dx, int incx, 
		     double *dy, int incy)
{
  int i;
  int n = *n_;
  if (n <= 0){ return; }
  if (incx == 1 && incy == 1){
    memcpy(dy, dx, sizeof(double) * ((unsigned) n));
  }else if (incx == 0 && incy == 1) {
    double x = dx[0];
    for (i = 0; i < n; ++i){ dy[i] = x; }
  }else {
    for (i = 0; i < n; ++i){ dy[i*incy] = dx[i*incx]; }
  }
} /* dcopy___ */

/* CONSTANT TIMES A VECTOR PLUS A VECTOR. */
// dy += da*dx
static void daxpy_sl__(int *n_, const double *da_, const double *dx, 
		       int incx, double *dy, int incy)
{
  int n = *n_, i;  
  double da = *da_;
  if (n <= 0 || da == 0){ return; }
  for (i = 0; i < n; ++i){ dy[i*incy] += da * dx[i*incx]; }
}

/* dot product dx dot dy. */
static double ddot_sl__(int *n_, double *dx, int incx, double *dy, int incy)
{
  int n = *n_;
  int i;
  long double sum = 0;
  if (n <= 0){ return 0; }
  for (i = 0; i < n; ++i){ sum += dx[i*incx] * dy[i*incy]; }
  return (double) sum;
}

/* compute the L2 norm of array DX of length N, stride INCX */
static double dnrm2___(int *n_, double *dx, int incx)
{
  int i;
  int n = *n_;
  double xmax = 0;
  double scale;
  long double sum = 0;
  for (i = 0; i < n; ++i) {
    double xabs = fabs(dx[incx*i]);
    if (xmax < xabs){ xmax = xabs; }
  }
  if (xmax == 0){ return 0; }
  scale = 1.0 / xmax;
  for (i = 0; i < n; ++i) {
    double xs = scale * dx[incx*i];
    sum += xs * xs;
  }
  return xmax * sqrt((double) sum);
}

/* apply Givens rotation */
// 1 <= i <= n までについて
// | dx[i] | = | c__ s_  | | dx[i] |
// | dy[i] |   | -s  c__ | | dy[i] |
// と計算
static void dsrot_(int n, double *dx, int incx, 
		   double *dy, int incy, double *c__, double *s_)
{
     int i;
     double c = *c__, s = *s_;
     for (i = 0; i < n; ++i) {
	  double x = dx[incx*i], y = dy[incy*i];
	  dx[incx*i] = c * x + s * y;
	  dy[incy*i] = c * y - s * x;
     }
}

/* construct Givens rotation */
// | da | に対して  |  c  s | | da |  = | roe |
// | db |         | -s  c | | db |    |   0  |
// となるような c, s を計算する。
// 計算後 da には roe の値,
// db には以下の値が入って返る.
// sigma =     1     if c = 0
//         sign(c)s  if |s| < |c|
//         sign(s)/c if |c| <= |s|
// このようにしておくと
// c = 0, s = 1                  if sigma = 0
// s = sigma, c = sqrt(1-s^2)    if |sigma | < 1
// c = 1/sigma, s = sqrt(1-c^2)  if |sigma | > 1
// として sigma から c,s を復元できる。
// c,s 二つを保持するよりも sigma だけ保持するほうがメモリ量としては有利ということらしい.
static void dsrotg_(double *da, double *db, double *c, double *s)
{
     double roe, scale;
     double absa = fabs(*da); double absb = fabs(*db);
     if (absa > absb) {
	  roe = *da; scale = absa;
     } else {
	  roe = *db; scale = absb;
     }

     if (scale != 0) {
       double iscale = 1 / scale;
       double tmpa = (*da) * iscale, tmpb = (*db) * iscale; // スケーリングした da と db
       // r の符号は da,db のうちで絶対値の大きいほうにあわせる。 r の絶対値は sqrt( | da^2 + db^2 |^2 ) に等しい.
       double r = (roe < 0 ? -scale : scale) * sqrt((tmpa * tmpa) + (tmpb * tmpb)); 
       *c = *da / r; *s = *db / r; 
       *da = r;
       if (*c != 0 && fabs(*c) <= *s){
	 *db = 1 / *c;
       }else{
	 *db = *s;
       }
     } else { 
	  *c = 1; 
	  *s = *da = *db = 0;
     }
}

/* scales vector X(n) by constant da */
static void dscal_sl__(int *n_, const double *da, double *dx, int incx)
{
  int n = *n_;
  double alpha = *da;
  int i;
  for (i = 0; i < n; ++i){ dx[i*incx] *= alpha; }
}

/**************************************************************************/

static const int c__0 = 0;
static const int c__1 = 1;
static const int c__2 = 2;

#define MIN2(a,b) ((a) <= (b) ? (a) : (b))
#define MAX2(a,b) ((a) >= (b) ? (a) : (b))

// ベクトル a を [ u[lpivot*iue], u[l1*iue], u[(l1+1)*iue], u[(l1+2)*iue], ..., u[m*iue] ]
// と表した際に
// P = I - (vv^t)/gamma
// Pv = [ -sign(u[lpivot])*sigma, 0, 0, 0, ..., 0 ]
// となる P を求めて、
// c__[lpivot,l1:m][0:ncv] に対して P　をかけて更新する。
// c は 
// | 1 5  9 13 |
// | 2 6 10 14 |
// | 3 7 11 15 |
// | 4 8 12 16 |
// のように保持されており, 5番目と6番目の要素のように縦には配列の要素として ice 分、間があり
// 10 番目と 14 番目の要素のように横には icv 分の間がある。
// mode = 1 のときは v と gamma の計算が行われるが、
// mode = 2 のときは v と gamma の情報は u と up に保持されているものとして
// c__[lpivot,l1:m][0:ncv] の更新のみ行われる。
// mode = 1 のときは up に sign(u[pivot]) * ( |u[pivot]| + sigma ),
//      u[lpivot*iue] に - sign(u[pivot]) * sigma の値が入る.
//
static void h12_(const int *mode,
		 const int *lpivot, const int *l1, const int *m,
		 double *u, const int *iue, double *up, 
		 double *c__, const int *ice, const int *icv, const int *ncv)
{
    /* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int u_dim1, u_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, j, i2, i3, i4;
    double sm;
    int incr;

/*     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUN 12 */
/*     TO APPEAR IN 'SOLVING LEAST SQUARES PROBLEMS', PRENTICE-HALL, 1974 */
/*     CONSTRUCTION AND/OR APPLICATION OF A SINGLE */
/*     HOUSEHOLDER TRANSFORMATION  Q = I + U*(U**T)/B */
/*     MODE    = 1 OR 2   TO SELECT ALGORITHM  H1  OR  H2 . */
/*     LPIVOT IS THE INDEX OF THE PIVOT ELEMENT. */
/*     L1,M   IF L1 <= M   THE TRANSFORMATION WILL BE CONSTRUCTED TO */
/*            ZERO ELEMENTS INDEXED FROM L1 THROUGH M. */
/*            IF L1 > M THE SUBROUTINE DOES AN IDENTITY TRANSFORMATION. */
/*     U(),IUE,UP */
/*            ON ENTRY TO H1 U() STORES THE PIVOT VECTOR. */
/*            IUE IS THE STORAGE INCREMENT BETWEEN ELEMENTS. */
/*            ON EXIT FROM H1 U() AND UP STORE QUANTITIES DEFINING */
/*            THE VECTOR U OF THE HOUSEHOLDER TRANSFORMATION. */
/*            ON ENTRY TO H2 U() AND UP */
/*            SHOULD STORE QUANTITIES PREVIOUSLY COMPUTED BY H1. */
/*            THESE WILL NOT BE MODIFIED BY H2. */
/*     C()    ON ENTRY TO H1 OR H2 C() STORES A MATRIX WHICH WILL BE */
/*            REGARDED AS A SET OF VECTORS TO WHICH THE HOUSEHOLDER */
/*            TRANSFORMATION IS TO BE APPLIED. */
/*            ON EXIT C() STORES THE SET OF TRANSFORMED VECTORS. */
/*     ICE    STORAGE INCREMENT BETWEEN ELEMENTS OF VECTORS IN C(). */
/*     ICV    STORAGE INCREMENT BETWEEN VECTORS IN C(). */
/*     NCV    NUMBER OF VECTORS IN C() TO BE TRANSFORMED. */
/*            IF NCV <= 0 NO OPERATIONS WILL BE DONE ON C(). */
    /* Parameter adjustments */
    u_dim1 = *iue;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    --c__;

    /* Function Body */
    if( 0 >= *lpivot || *lpivot >= *l1 || *l1 > *m ){
      return;
    }

    double cl = fabs( u[*lpivot * u_dim1 + 1] );
    if( *mode == 2 ){
      if( cl <= 0.0 ){ return; }
    } else {
      
      /* ****** CONSTRUCT THE TRANSFORMATION ****** */
      for( j = *l1; j <= *m; ++j ){
	cl = MAX2( cl, fabs(u[j * u_dim1 + 1]) );
      }
      if( cl <= 0.0 ){ return; }
      
      double clinv = one / cl; 
      // sm に二乗和を設定する。
      d__1 = u[*lpivot * u_dim1 + 1] * clinv;
      sm = d__1 * d__1;
      for( j = *l1; j <= *m; ++j ){
	d__1 = u[j * u_dim1 + 1] * clinv;
	sm += d__1 * d__1;
      }
      cl *= sqrt(sm);
      if( u[*lpivot * u_dim1 + 1] > 0.0 ){
	cl = -cl;
      }
      *up = u[*lpivot * u_dim1 + 1] - cl; // sign(u[pivot]) * ( |u[pivot]| + sigma )
      u[*lpivot * u_dim1 + 1] = cl; // - sign(u[pivot]) * sigma 
    }
    
    /* ****** APPLY THE TRANSFORMATION  I+U*(U**T)/B  TO C ****** */
    if( *ncv <= 0 ){ return; }
    
    double b = (*up) * u[*lpivot * u_dim1 + 1];
    if( b >= 0.0 ){ return ; }
    b = one / b;

    i2 = 1 - *icv + *ice * (*lpivot - 1); // C(ipivot,1) からインデックスを開始するための設定.
    incr = *ice * (*l1 - *lpivot);
    for( j = 1; j <= *ncv; ++j ){
        i2 += *icv;     // C(ipivot, j)
	i3 = i2 + incr; // C(l1, j)
	i4 = i3;        
	sm = c__[i2] * (*up);
	for( i__ = *l1; i__ <= *m; ++i__ ){
	    sm += c__[i3] * u[i__ * u_dim1 + 1];
	    i3 += *ice;
	}
	if (sm == 0.0) {
	  continue;
	}
	sm *= b; 
	c__[i2] += sm * (*up); // C(ipivot,j) += C(l1, j) + ... + C(m, j) .
	for( i__ = *l1; i__ <= *m; ++i__ ){
	    c__[i4] += sm * u[i__ * u_dim1 + 1];
	    i4 += *ice;
	}
    }

    return;
} /* h12_ */


static void set_wmax( double *wmax_p, int *izmax_p,
		      int iz1, int iz2,
		      const double *w, const int *indx )
{
  for( int iz = iz1; iz <= iz2; ++iz ){
    int j = indx[iz];
    if( w[j] > *wmax_p ){
      *wmax_p = w[j];
      *izmax_p = iz;
    }
  }
}

// a は column wize に保持.
static void nnls_(double *a, int *mda, int *m, int *n, 
		  double *b, double *x, double *rnorm,
		  double *w, double *z__, int *indx, int *mode)
{
    /* Initialized data */

    const double one = 1.;
    const double factor = .01;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    double c__;
    int i__, j, k, l;
    double s, t;
    int ii, jj, ip, iz, jz;
    double up;
    int iz1, iz2, npp1, iter;
    double wmax, alpha, asave;
    int itmax, izmax, nsetp;
    double unorm;

    // 以下のアルゴリズムは以下の url のページに概要が書いてある,が
    // SOLVING LEAST SQUARES PROBLEMS のほうがそのままかいてあるのでわかりやすい..
    // https://en.wikipedia.org/wiki/Non-negative_least_squares

    
/*     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY: */
/*     'SOLVING LEAST SQUARES PROBLEMS'. PRENTICE-HALL.1974 */
/*      **********   NONNEGATIVE LEAST SQUARES   ********** */
/*     GIVEN AN M BY N MATRIX, A, AND AN M-VECTOR, B, COMPUTE AN */
/*     N-VECTOR, X, WHICH SOLVES THE LEAST SQUARES PROBLEM */
/*                  A*X = B  SUBJECT TO  X >= 0 */
/*     A(),MDA,M,N */
/*            MDA IS THE FIRST DIMENSIONING PARAMETER FOR THE ARRAY,A(). */
/*            ON ENTRY A()  CONTAINS THE M BY N MATRIX,A. */
/*            ON EXIT A() CONTAINS THE PRODUCT Q*A, */
/*            WHERE Q IS AN M BY M ORTHOGONAL MATRIX GENERATED */
/*            IMPLICITLY BY THIS SUBROUTINE. */
/*            EITHER M>=N OR M<N IS PERMISSIBLE. */
/*            THERE IS NO RESTRICTION ON THE RANK OF A. */
/*     B()    ON ENTRY B() CONTAINS THE M-VECTOR, B. */
/*            ON EXIT B() CONTAINS Q*B. */
/*     X()    ON ENTRY X() NEED NOT BE INITIALIZED. */
/*            ON EXIT X() WILL CONTAIN THE SOLUTION VECTOR. */
/*     RNORM  ON EXIT RNORM CONTAINS THE EUCLIDEAN NORM OF THE */
/*            RESIDUAL VECTOR. */
/*     W()    AN N-ARRAY OF WORKING SPACE. */
/*            ON EXIT W() WILL CONTAIN THE DUAL SOLUTION VECTOR. */
/*            W WILL SATISFY W(I)=0 FOR ALL I IN SET P */
/*            AND W(I)<=0 FOR ALL I IN SET Z */
/*     Z()    AN M-ARRAY OF WORKING SPACE. */
/*     INDX()AN INT WORKING ARRAY OF LENGTH AT LEAST N. */
/*            ON EXIT THE CONTENTS OF THIS ARRAY DEFINE THE SETS */
/*            P AND Z AS FOLLOWS: */
/*            INDX(1)    THRU INDX(NSETP) = SET P. */
/*            INDX(IZ1)  THRU INDX (IZ2)  = SET Z. */
/*            IZ1=NSETP + 1 = NPP1, IZ2=N. */
/*     MODE   THIS IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANING: */
/*            1    THE SOLUTION HAS BEEN COMPUTED SUCCESSFULLY. */
/*            2    THE DIMENSIONS OF THE PROBLEM ARE WRONG, */
/*                 EITHER M <= 0 OR N <= 0. */
/*            3    ITERATION COUNT EXCEEDED, MORE THAN 3*N ITERATIONS. */
    /* Parameter adjustments */
    --z__;
    --b;
    --indx;
    --w;
    --x;
    a_dim1 = *mda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    /*     revised          Dieter Kraft, March 1983 */
    if (*m <= 0 || *n <= 0) {
      *mode = 2;
      return;
    }
    
    *mode = 1;
    iter = 0;
    itmax = *n * 3;

    /* STEP ONE (INITIALIZE) */

    // indx[iz1:iz2] は 集合 Z をあらわす。
    iz1 = 1; iz2 = *n;
    for (i__ = 1; i__ <= *n; ++i__) {
	indx[i__] = i__;
    }
    
    nsetp = 0;
    npp1 = 1;

    // inittialize x[:] = 0.0
    x[1] = 0.0;
    dcopy___(n, &x[1], 0, &x[1], 1);
    
    /* STEP TWO (COMPUTE DUAL VARIABLES) */
    /* .....ENTRY LOOP A */
L110:
    if (iz1 > iz2 || nsetp >= *m) { goto L280; }

    // calc w = A^t * b
    // w = A^t * ( b - A*x ) の計算だが b は b - A*x のように更新されていくのでこれでよい.
    for (iz = iz1; iz <= iz2; ++iz) {
	j = indx[iz];
	i__2 = *m - nsetp;
	w[j] = ddot_sl__(&i__2, &a[npp1 + j * a_dim1], 1, &b[npp1], 1);
    }
    
    /* STEP THREE (TEST DUAL VARIABLES) */
    while(1){
      wmax = 0.0;
      set_wmax( &wmax, &izmax, iz1, iz2, w, indx );
      /* .....EXIT LOOP A */
      if( wmax <= 0.0 ){ goto L280; }
      
      iz = izmax;
      j = indx[iz];
      /* STEP FOUR (TEST INDX J FOR LINEAR DEPENDENCY) */
      asave = a[npp1 + j * a_dim1]; // h12_ で書き換わるので保持しておき、更新が終わったら元に戻す.
      i__2 = npp1 + 1;
      h12_(&c__1,
	   &npp1, &i__2, m,
	   &a[j * a_dim1 + 1], &c__1, &up,
	   &z__[1], &c__1, &c__1, &c__0); // ここでは H を設定するだけで, z の更新はしない.
      unorm = dnrm2___(&nsetp, &a[j * a_dim1 + 1], 1);
      t = factor * fabs(a[npp1 + j * a_dim1]);
      d__1 = unorm + t; // a[0] + sigma
      if( d__1 - unorm <= 0.0 ){
	;
      }else{
	dcopy___(m, &b[1], 1, &z__[1], 1); // z[:] = b[:].
	i__2 = npp1 + 1;
	// z = H*z 
	h12_(&c__2,
	     &npp1, &i__2, m,
	     &a[j * a_dim1 + 1], &c__1, &up,
	     &z__[1], &c__1, &c__1, &c__1);
	if (z__[npp1] / a[npp1 + j * a_dim1] > 0.0) {
	  break;
	}
      }
      a[npp1 + j * a_dim1] = asave;
      w[j] = 0.0;
    }

    /* STEP FIVE (ADD COLUMN) */
    dcopy___(m, &z__[1], 1, &b[1], 1);

    // indx[iz] のところにあった j が Z(indx) から pop される。
    indx[iz] = indx[iz1]; 
    indx[iz1] = j;
    ++iz1;
    
    nsetp = npp1;
    ++npp1;
    for (jz = iz1; jz <= iz2; ++jz) {
	jj = indx[jz];
	h12_(&c__2,
	     &nsetp, &npp1, m,
	     &a[j * a_dim1 + 1], &c__1, &up, 
	     &a[jj * a_dim1 + 1], &c__1, mda, &c__1);
    }
    k = MIN2( npp1, *mda );
    w[j] = 0.0;
    i__2 = *m - nsetp;
    dcopy___(&i__2, &w[j], 0, &a[k + j * a_dim1], 1);

    /* STEP SIX (SOLVE LEAST SQUARES SUB-PROBLEM) */
    /* .....ENTRY LOOP B */
L180:
    // P にインデックスのあるカラムから作られる行列については、
    // 上のほうで上三角に変換してあるので,普通に解いていくことができる.
    // z に解が入る。
    for (ip = nsetp; ip >= 1; --ip) { 
	if (ip == nsetp) {
	    ;
	} else {
	  d__1 = -z__[ip + 1];
	  daxpy_sl__(&ip, &d__1, &a[jj * a_dim1 + 1], 1, &z__[1], 1);
	}
	jj = indx[ip];
	z__[ip] /= a[ip + jj * a_dim1];
    }
    ++iter;
    if (iter <= itmax) {
      ;
    }else {
      *mode = 3;
      goto L280;
    }
    
    /* STEP SEVEN TO TEN (STEP LENGTH ALGORITHM) */

    // P の要素 q =indx[ip] について
    // x[indx[ip]]/(x[indx[ip]] - z[ip]) が最小となる時の ip を jj に、
    // x[q]/(x[q] - z[ip]) の値自体を alpha に保持する。
    alpha = one;
    jj = 0;
    for (ip = 1; ip <= nsetp; ++ip) {
	if (z__[ip] > 0.0) { continue; }
	l = indx[ip];
	t = -x[l] / (z__[ip] - x[l]);
	if (alpha < t) { continue; }
	alpha = t;
	jj = ip;
    }
    
    // x = x + alpha * ( z - x ) で更新
    for (ip = 1; ip <= nsetp; ++ip) {
	l = indx[ip];
	x[l] = (one - alpha) * x[l] + alpha * z__[ip];
    }
    /* .....EXIT LOOP B */
    if (jj == 0) { goto L110; }
    
    
    /* STEP ELEVEN (DELETE COLUMN) */
    i__ = indx[jj];
L250:
    x[i__] = 0.0;
    ++jj;
    // jj を追い出す.
    // indx[jj+1:nstep] までを ndx[jj:nstep-1 ] につめなおす.
    // それに伴って E_P が
    // | a a a a a |      | a a a a |
    // |   a a a a |      |   a a a |
    // |     a a a | ->   |     a a |
    // |       a a |      |     a a |
    // |         a |      |       a |
    // のように上三角でなくなるのでその分を修正して、また上三角になるようにする。
    for( j = jj; j <= nsetp; ++j ){
      ii = indx[j]; 
      indx[j - 1] = ii;
      dsrotg_( &a[j - 1 + ii * a_dim1], &a[j + ii * a_dim1], &c__, &s );
      t = a[j - 1 + ii * a_dim1];
      dsrot_(*n, &a[j - 1 + a_dim1], *mda, &a[j + a_dim1], *mda, &c__, &s);
      a[j - 1 + ii * a_dim1] = t;
      a[j + ii * a_dim1] = 0.0;
      dsrot_(1, &b[j - 1], 1, &b[j], 1, &c__, &s);
    }
    npp1 = nsetp;
    --nsetp;
    --iz1;
    indx[iz1] = i__;
    if( nsetp <= 0 ){
      *mode = 3;
      goto L280;
    }
    for( jj = 1; jj <= nsetp; ++jj ){
	i__ = indx[jj];
	if( x[i__] <= 0.0 ){
	    goto L250;
	}
    }
    dcopy___(m, &b[1], 1, &z__[1], 1);
    goto L180;
    
    /* STEP TWELVE (SOLUTION) */
L280:
    k = MIN2( npp1, *m );
    i__2 = *m - nsetp;
    *rnorm = dnrm2___(&i__2, &b[k], 1);
    if (npp1 > *m) {
	w[1] = 0.0;
	dcopy___(n, &w[1], 0, &w[1], 1);
    }
    
    /* END OF SUBROUTINE NNLS */
    return;
} /* nnls_ */


static void ldp_(double *g, int *mg, int *m, int *n, 
		 double *h__, double *x, double *xnorm,
		 double *w, int *indx, int *mode)
{
    /* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int g_dim1, g_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, j, n1, if__, iw, iy, iz;
    double fac;
    double rnorm;
    int iwdual;

    // ldp を nnls をもちいて解いている
    // どのようにされているかは
    //     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY: 
    //    'SOLVING LEAST SQUARES PROBLEMS'. PRENTICE-HALL.1974 
    // の　chapter23, section 4 problem ldp をみるとそのまま書いてある。

/*                     T */
/*     MINIMIZE   (1/2)*X*X    SUBJECT TO   G * X >= H. */
/*       C.L. LAWSON, R.J. HANSON: 'SOLVING LEAST SQUARES PROBLEMS' */
/*       PRENTICE HALL, ENGLEWOOD CLIFFS, NEW JERSEY, 1974. */
/*     PARAMETER DESCRIPTION: */
/*     G(),MG,M,N   ON ENTRY G() STORES THE M BY N MATRIX OF */
/*                  LINEAR INEQUALITY CONSTRAINTS. G() HAS FIRST */
/*                  DIMENSIONING PARAMETER MG */
/*     H()          ON ENTRY H() STORES THE M VECTOR H REPRESENTING */
/*                  THE RIGHT SIDE OF THE INEQUALITY SYSTEM */
/*     REMARK: G(),H() WILL NOT BE CHANGED DURING CALCULATIONS BY LDP */
/*     X()          ON ENTRY X() NEED NOT BE INITIALIZED. */
/*                  ON EXIT X() STORES THE SOLUTION VECTOR X IF MODE=1. */
/*     XNORM        ON EXIT XNORM STORES THE EUCLIDIAN NORM OF THE */
/*                  SOLUTION VECTOR IF COMPUTATION IS SUCCESSFUL */
/*     W()          W IS A ONE DIMENSIONAL WORKING SPACE, THE LENGTH */
/*                  OF WHICH SHOULD BE AT LEAST (M+2)*(N+1) + 2*M */
/*                  ON EXIT W() STORES THE LAGRANGE MULTIPLIERS */
/*                  ASSOCIATED WITH THE CONSTRAINTS */
/*                  AT THE SOLUTION OF PROBLEM LDP */
/*     INDX()      INDX() IS A ONE DIMENSIONAL INT WORKING SPACE */
/*                  OF LENGTH AT LEAST M */
/*     MODE         MODE IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING */
/*                  MEANINGS: */
/*          MODE=1: SUCCESSFUL COMPUTATION */
/*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N.LE.0) */
/*               3: ITERATION COUNT EXCEEDED BY NNLS */
/*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
    /* Parameter adjustments */
    --indx;
    --h__;
    --x;
    g_dim1 = *mg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    --w;

    /* Function Body */

    if (*n <= 0) {
      *mode = 2;
      return;
    }
    
    /*  STATE DUAL PROBLEM */
    *mode = 1;
    
    // x[:] = 0.0 で初期化.
    x[1] = 0.0;
    dcopy___(n, &x[1], 0, &x[1], 1);
    *xnorm = 0.0;
    if (*m == 0) { return; }

    // set E = (G,h)T
    iw = 0;
    for (j = 1; j <= *m; ++j) {
	for (i__ = 1; i__ <= *n; ++i__) {
	    ++iw;
	    w[iw] = g[j + i__ * g_dim1];
	}
	++iw;
	w[iw] = h__[j];
    }

    // set f = (0,0,....,0,1)T
    if__ = iw + 1;
    for (i__ = 1; i__ <= *n; ++i__) {
	++iw;
	w[iw] = 0.0;
    }
    w[iw + 1] = one;
    
    n1 = *n + 1;
    iz = iw + 2;
    iy = iz + n1;
    iwdual = iy + *m;
    /*  SOLVE DUAL PROBLEM */
    // w[1:] = E
    // w[if__:] = f
    // w[iy:] = u
    // w[iwdual:] : nnls_ のワークスペース
    // w[iz:] : nnls_ のワークスペース
    // indx : nnls_ のワークスペース
    nnls_( &w[1], &n1, &n1, m,
	   &w[if__], &w[iy], &rnorm,
	   &w[iwdual], &w[iz], &indx[1], mode);
    if (*mode != 1) { return; }

    *mode = 4;
    if (rnorm <= 0.0) { return; }

    /*  COMPUTE SOLUTION OF PRIMAL PROBLEM */
    // fac =  - r_(n+1)
    fac = one - ddot_sl__(m, &h__[1], 1, &w[iy], 1);

    // check fac <= 0, 数値誤差を気にして 1 足して比較している？
    d__1 = one + fac;
    if (d__1 - one <= 0.0) { return; }
    
    *mode = 1;
    // set x
    fac = one / fac; // fac = 1/(-r_(n+1))
    for (j = 1; j <= *n; ++j) {
	x[j] = fac * ddot_sl__(m, &g[j * g_dim1 + 1], 1, &w[iy], 1);
    }
    *xnorm = dnrm2___(n, &x[1], 1);
    
    /*  COMPUTE LAGRANGE MULTIPLIERS FOR PRIMAL PROBLEM */
    // w = u / (-r_(n+1)), これが lagrange 乗数になる.
    w[1] = 0.0;
    dcopy___(m, &w[1], 0, &w[1], 1);
    daxpy_sl__(m, &fac, &w[iy], 1, &w[1], 1);
    
    /*  END OF SUBROUTINE LDP */
    return;
} /* ldp_ */

static void lsi_(double *e, double *f,
		 double *g, double *h__,
		 int *le, int *me, int *lg, int *mg, 
		 int *n, double *x, double *xnorm, 
		 double *w, int *jw, int *mode)
{
    /* Initialized data */

    const double epmach = 2.22e-16;
    const double one = 1.;

    /* System generated locals */
    int e_dim1, e_offset, g_dim1, g_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j;
    double t;

    // lsi を ldp に変換して解いている.
    // 変換がどのようにされているかは
    //     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY: 
    //    'SOLVING LEAST SQUARES PROBLEMS'. PRENTICE-HALL.1974 
    // の　chapter23, section 5 converting problem lsi to problem ldp をみるとそのまま書いてある。
    
/*     FOR MODE=1, THE SUBROUTINE RETURNS THE SOLUTION X OF */
/*     INEQUALITY CONSTRAINED LINEAR LEAST SQUARES PROBLEM: */
/*                    MIN ||E*X-F|| */
/*                     X */
/*                    S.T.  G*X >= H */
/*     THE ALGORITHM IS BASED ON QR DECOMPOSITION AS DESCRIBED IN */
/*     CHAPTER 23.5 OF LAWSON & HANSON: SOLVING LEAST SQUARES PROBLEMS */
/*     THE FOLLOWING DIMENSIONS OF THE ARRAYS DEFINING THE PROBLEM */
/*     ARE NECESSARY */
/*     DIM(E) :   FORMAL (LE,N),    ACTUAL (ME,N) */
/*     DIM(F) :   FORMAL (LE  ),    ACTUAL (ME  ) */
/*     DIM(G) :   FORMAL (LG,N),    ACTUAL (MG,N) */
/*     DIM(H) :   FORMAL (LG  ),    ACTUAL (MG  ) */
/*     DIM(X) :   N */
/*     DIM(W) :   (N+1)*(MG+2) + 2*MG */
/*     DIM(JW):   LG */
/*     ON ENTRY, THE USER HAS TO PROVIDE THE ARRAYS E, F, G, AND H. */
/*     ON RETURN, ALL ARRAYS WILL BE CHANGED BY THE SUBROUTINE. */
/*     X     STORES THE SOLUTION VECTOR */
/*     XNORM STORES THE RESIDUUM OF THE SOLUTION IN EUCLIDIAN NORM */
/*     W     STORES THE VECTOR OF LAGRANGE MULTIPLIERS IN ITS FIRST */
/*           MG ELEMENTS */
/*     MODE  IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANINGS: */
/*          MODE=1: SUCCESSFUL COMPUTATION */
/*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N<1) */
/*               3: ITERATION COUNT EXCEEDED BY NNLS */
/*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
/*               5: MATRIX E IS NOT OF FULL RANK */
/*     03.01.1980, DIETER KRAFT: CODED */
/*     20.03.1987, DIETER KRAFT: REVISED TO FORTRAN 77 */
    /* Parameter adjustments */
    --f;
    --jw;
    --h__;
    --x;
    g_dim1 = *lg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    e_dim1 = *le;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    --w;

    /* Function Body */
    /*  QR-FACTORS OF E AND APPLICATION TO F */
    for( i__ = 1; i__ <= *n; ++i__ ){
	i__2 = i__ + 1;
	j = MIN2( i__2, *n );
	i__3 = *n - i__;
	h12_(&c__1,
	     &i__, &i__2, me,
	     &e[i__ * e_dim1 + 1], &c__1, &t, 
	     &e[j * e_dim1 + 1], &c__1, le, &i__3);
	h12_(&c__2,
	     &i__, &i__2, me,
	     &e[i__ * e_dim1 + 1], &c__1, &t,
	     &f[1], &c__1, &c__1, &c__1);
    }
    /*  TRANSFORM G AND H TO GET LEAST DISTANCE PROBLEM */
    for( i__ = 1; i__ <= *mg; ++i__ ){
	for( j = 1; j <= *n; ++j ){
	  if( fabs(e[j + j * e_dim1]) < epmach ){
	      *mode = 5;
	      return;
	    }
	    i__3 = j - 1;
	    g[i__ + j * g_dim1] = (g[i__ + j * g_dim1] 
				   - ddot_sl__(&i__3, &g[i__ + g_dim1], 
					       *lg, &e[j * e_dim1 + 1], 1)) 
	                           / e[j + j *e_dim1];
	}
	h__[i__] -= ddot_sl__(n, &g[i__ + g_dim1], *lg, &f[1], 1);
    }
    /*  SOLVE LEAST DISTANCE PROBLEM */
    ldp_(&g[g_offset], lg, mg, n,
	 &h__[1], &x[1], xnorm,
	 &w[1], &jw[1], mode);
    if( *mode != 1 ){ return; }

    /*  SOLUTION OF ORIGINAL PROBLEM */
    daxpy_sl__(n, &one, &f[1], 1, &x[1], 1);
    for (i__ = *n; i__ >= 1; --i__) {
	j = MIN2( i__ + 1, *n );
	i__2 = *n - i__;
	x[i__] = (x[i__] - ddot_sl__(&i__2, &e[i__ + j * e_dim1], *le, &x[j], 1))
	         / e[i__ + i__ * e_dim1];
    }

    j = MIN2( *n + 1, *me );
    i__2 = *me - *n;
    t = dnrm2___(&i__2, &f[j], 1);
    *xnorm = sqrt(*xnorm * *xnorm + t * t);
    /*  END OF SUBROUTINE LSI */

    return;
} /* lsi_ */


static void calc_h_and_lmax( int j, int m, int n,
			     double* a, int a_dim1, 
			     double* h__, int *lmax_p, double *hmax_p )
{
  *lmax_p = j;
  for( int l = j; l <= n; ++l ){ // 列番号
    h__[l] = 0.0;
    for( int i = j; i <= m; ++i ){ // 行番号
      double d__1 = a[ i + l*a_dim1 ];
      h__[l] += d__1 * d__1;
    }
    if( h__[l] > h__[*lmax_p] ){
      *lmax_p = l;
    }
  }
  *hmax_p = h__[*lmax_p];
  return;
}

//
// a : m*n の行列, 列 wize に保持し, ひとつの要素の間は mda 分離れている
// b : m*nb の行列, 列 wize に保持し、ひとつの要素は mdb 分離れている。
// tau : 対角要素がこの値以下の場合にはランク落ちしていると判断するためのクライテリア.
// krank : A の擬似ランクがこの変数に出力される。
// rnorm : 各 1<j<nb に対して || Ax -b[j] || の最終値が rnorm[j] に出力される。
// h__, g, ip : ワークスペース. len = n
static void hfti_(double *a, int *mda,
		  int *m, int *n, 
		  double *b, int *mdb, const int *nb,
		  double *tau, int *krank, double *rnorm,
		  double *h__, double *g, int *ip )
{
    /* Initialized data */

    const double factor = .001;

    /* System generated locals */
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j, k, l;
    int jb, kp1;
    double tmp, hmax;
    int lmax, ldiag;

/*     RANK-DEFICIENT LEAST SQUARES ALGORITHM AS DESCRIBED IN: */
/*     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUN 12 */
/*     TO APPEAR IN 'SOLVING LEAST SQUARES PROBLEMS', PRENTICE-HALL, 1974 */ // p.81 アルゴリズムの説明あり.
/*     A(*,*),MDA,M,N   THE ARRAY A INITIALLY CONTAINS THE M x N MATRIX A */
/*                      OF THE LEAST SQUARES PROBLEM AX = B. */
/*                      THE FIRST DIMENSIONING PARAMETER MDA MUST SATISFY */
/*                      MDA >= M. EITHER M >= N OR M < N IS PERMITTED. */
/*                      THERE IS NO RESTRICTION ON THE RANK OF A. */
/*                      THE MATRIX A WILL BE MODIFIED BY THE SUBROUTINE. */
/*     B(*,*),MDB,NB    IF NB = 0 THE SUBROUTINE WILL MAKE NO REFERENCE */
/*                      TO THE ARRAY B. IF NB > 0 THE ARRAY B() MUST */
/*                      INITIALLY CONTAIN THE M x NB MATRIX B  OF THE */
/*                      THE LEAST SQUARES PROBLEM AX = B AND ON RETURN */
/*                      THE ARRAY B() WILL CONTAIN THE N x NB SOLUTION X. */
/*                      IF NB>1 THE ARRAY B() MUST BE DOUBLE SUBSCRIPTED */
/*                      WITH FIRST DIMENSIONING PARAMETER MDB>=MAX(M,N), */
/*                      IF NB=1 THE ARRAY B() MAY BE EITHER SINGLE OR */
/*                      DOUBLE SUBSCRIPTED. */
/*     TAU              ABSOLUTE TOLERANCE PARAMETER FOR PSEUDORANK */
/*                      DETERMINATION, PROVIDED BY THE USER. */
/*     KRANK            PSEUDORANK OF A, SET BY THE SUBROUTINE. */
/*     RNORM            ON EXIT, RNORM(J) WILL CONTAIN THE EUCLIDIAN */
/*                      NORM OF THE RESIDUAL VECTOR FOR THE PROBLEM */
/*                      DEFINED BY THE J-TH COLUMN VECTOR OF THE ARRAY B. */
/*     H(), G()         ARRAYS OF WORKING SPACE OF LENGTH >= N. */
/*     IP()             INT ARRAY OF WORKING SPACE OF LENGTH >= N */
/*                      RECORDING PERMUTATION INDICES OF COLUMN VECTORS */
    /* Parameter adjustments */
    --ip;
    --g;
    --h__;
    a_dim1 = *mda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --rnorm;
    b_dim1 = *mdb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    k = 0;
    ldiag = MIN2(*m,*n);
    if( ldiag <= 0 ){
      *krank = k;
      return;
    }

    // h__[j] は A の j 列目 A[i:,j] に対して ||A[i:,j]||^2 の値を保持する。
    // A[i:,j] は j 列目の i 成分以降を要素とするベクトル.
    // この h の値が大きい列を pivot として householder 変換を行っていく.
    
    /*   COMPUTE LMAX */
    for( j = 1; j <= ldiag; ++j ){ //行番号かつ列番号。処理は対角成分を軸にすすんでいくため。
	if( j == 1 ){
	  calc_h_and_lmax( 1, *m, *n,
			   a, a_dim1, 
			   h__, &lmax, &hmax );
	}else{
	  // ここは 2 行目から実行されるが, h は 1 行目の実行で初期化されているものとしている.
	  lmax = j;
	  for( l = j; l <= *n; ++l ){ // 列番号
	    d__1 = a[j - 1 + l * a_dim1];
	    h__[l] -= d__1 * d__1;
	    if( h__[l] > h__[lmax] ){
		lmax = l;
	    }
	  }
	  d__1 = hmax + factor * h__[lmax];
	  if( d__1 - hmax <= 0.0 ){ // 残りの列の数値がそれまでより圧倒的に小さい場合にここに入り、 h を設定しなおす.
	    calc_h_and_lmax( j, *m, *n,
			     a, a_dim1, 
			     h__, &lmax, &hmax );
	  }
	}

	/*   COLUMN INTERCHANGES IF NEEDED */
	// norm が max な列と交換.
	ip[j] = lmax;
	if( j != lmax ){
	  for (i__ = 1; i__ <= *m; ++i__) { 
	    tmp = a[i__ + j * a_dim1];
	    a[i__ + j * a_dim1] = a[i__ + lmax * a_dim1];
	    a[i__ + lmax * a_dim1] = tmp;
	  }
	  h__[lmax] = h__[j];
	}

	/*   J-TH TRANSFORMATION AND APPLICATION TO A AND B */
	i__ = MIN2( j + 1, *n );
	i__2 = j + 1;
	i__3 = *n - j;
	// A[j:,j:] を考えて、 最初の列を pivot として householder 変換をする。
	h12_(&c__1,
	     &j, &i__2, m,
	     &a[j * a_dim1 + 1], &c__1, &h__[j], 
	     &a[i__ *a_dim1 + 1], &c__1, mda, &i__3);
	// b にも同じ householder 変換をする。
	h12_(&c__2,
	     &j, &i__2, m,
	     &a[j * a_dim1 + 1], &c__1, &h__[j], 
	     &b[b_offset], &c__1, mdb, nb);
    } // end : for( j = 1; j <= ldiag; ++j )
    
    /*   DETERMINE PSEUDORANK */
    for( j = 1; j <= ldiag; ++j ){
      if( fabs(a[j + j * a_dim1]) <= *tau ){
	k = j - 1;
	goto L110;
      }
    }
    k = ldiag;
    
L110:
    kp1 = k + 1;
    /*   NORM OF RESIDUALS */
    // || Ax - b || の計算.
    for( jb = 1; jb <= *nb; ++jb ){
	i__1 = *m - k;
	rnorm[jb] = dnrm2___(&i__1, &b[kp1 + jb * b_dim1], 1);
    }
    
    if( k == 0 ){
      // A のランクが 0, つまり,A のすべての要素が小さすぎる。。
      // この場合、答えはすべて 0 として返す。
      for( jb = 1; jb <= *nb; ++jb ){
	for( i__ = 1; i__ <= *n; ++i__ ){
	  b[i__ + jb * b_dim1] = 0.0;
	}
      }
      *krank = 0;
      return;
    }

    if( k != *n ){
      // もともと n > m な行列、もしくは rank 落ちした場合.
      /*   HOUSEHOLDER DECOMPOSITION OF FIRST K ROWS */
      // | A_11 A_12 | -> | R_11 R_12 | , (R_11 が full_rank ) のように分解された場合.
      // | A_21 A_22 |    |    0 R_22 |
      // A_21 の部分の消去演算
      for( i__ = k; i__ >= 1; --i__ ){
	i__2 = i__ - 1;
	h12_(&c__1,
	     &i__, &kp1, n,
	     &a[i__ + a_dim1], mda, &g[i__], 
	     &a[a_offset], mda, &c__1, &i__2 );
      }
    }
    
    for( jb = 1; jb <= *nb; ++jb ) { // b の列番号.
      /*   SOLVE K*K TRIANGULAR SYSTEM */
	for( i__ = k; i__ >= 1; --i__ ){
	  j = MIN2( i__ + 1, *n );
	  i__1 = k - i__;
	  b[i__ + jb * b_dim1] = ( b[i__ + jb * b_dim1] 
				   - ddot_sl__(&i__1, &a[i__ + j * a_dim1], 
					       *mda, &b[j + jb * b_dim1], 1)) 
	                          / a[i__ + i__ * a_dim1];
	}
	/*   COMPLETE SOLUTION VECTOR */
	if( k != *n ){
	  // もともと n > m な行列、もしくは rank 落ちした場合.
	  // | A_11 A_12 | -> | R_11 R_12 | , (R_11 が full_rank ) のように分解された場合.
	  // | A_21 A_22 |    |    0 R_22 |
	  for( j = kp1; j <= *n; ++j ){ 
	    b[j + jb * b_dim1] = 0.0;
	  }
	  for( i__ = 1; i__ <= k; ++i__ ){ // A_21 成分の消去のための householder 変換をを b にも行う。
	    h12_(&c__2,
		 &i__, &kp1, n,
		 &a[i__ + a_dim1], mda, &g[i__],
		 &b[jb * b_dim1 + 1], &c__1, mdb, &c__1);
	  }
	}

	/*   REORDER SOLUTION ACCORDING TO PREVIOUS COLUMN INTERCHANGES */
	// 上の方で pivot 変換をおこなっているため、修正する。
	for( j = ldiag; j >= 1; --j ){
	    if( ip[j] == j ){ continue; }
	    l = ip[j];
	    tmp = b[l + jb * b_dim1];
	    b[l + jb * b_dim1] = b[j + jb * b_dim1];
	    b[j + jb * b_dim1] = tmp;
	}
    }
    
    *krank = k;
    return;
} /* hfti_ */

//
static void lsei_(double *c__, double *d__, 
		  double *e, double *f, 
		  double *g, double *h__, 
		  int *lc, int *mc, 
		  int *le, int *me, 
		  int *lg, int *mg, 
		  int *n, 
		  double *x, double *xnrm, 
		  double *w, int *jw, int *mode)
{
    /* Initialized data */

    const double epmach = 2.22e-16;

    /* System generated locals */
    int c_dim1, c_offset, e_dim1, e_offset, g_dim1, g_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j, k, l;
    double t;
    int ie, if__, ig, iw, mc1;
    int krank;

/*     FOR MODE=1, THE SUBROUTINE RETURNS THE SOLUTION X OF */
/*     EQUALITY & INEQUALITY CONSTRAINED LEAST SQUARES PROBLEM LSEI : */
/*                MIN ||E*X - F|| */
/*                 X */
/*                S.T.  C*X  = D, */
/*                      G*X >= H. */
/*     USING QR DECOMPOSITION & ORTHOGONAL BASIS OF NULLSPACE OF C */
/*     CHAPTER 23.6 OF LAWSON & HANSON: SOLVING LEAST SQUARES PROBLEMS. */
/*     THE FOLLOWING DIMENSIONS OF THE ARRAYS DEFINING THE PROBLEM */
/*     ARE NECESSARY */
/*     DIM(E) :   FORMAL (LE,N),    ACTUAL (ME,N) */ // store 方法は FORMAL の方にあわせているが計算上は ACTUAL の法を参考にする.
/*     DIM(F) :   FORMAL (LE  ),    ACTUAL (ME  ) */
/*     DIM(C) :   FORMAL (LC,N),    ACTUAL (MC,N) */
/*     DIM(D) :   FORMAL (LC  ),    ACTUAL (MC  ) */
/*     DIM(G) :   FORMAL (LG,N),    ACTUAL (MG,N) */
/*     DIM(H) :   FORMAL (LG  ),    ACTUAL (MG  ) */
/*     DIM(X) :   FORMAL (N   ),    ACTUAL (N   ) */
/*     DIM(W) :   2*MC+ME+(ME+MG)*(N-MC)  for LSEI */
/*              +(N-MC+1)*(MG+2)+2*MG     for LSI */
/*     DIM(JW):   MAX(MG,L) */
/*     ON ENTRY, THE USER HAS TO PROVIDE THE ARRAYS C, D, E, F, G, AND H. */
/*     ON RETURN, ALL ARRAYS WILL BE CHANGED BY THE SUBROUTINE. */
/*     X     STORES THE SOLUTION VECTOR */
/*     XNORM STORES THE RESIDUUM OF THE SOLUTION IN EUCLIDIAN NORM */
/*     W     STORES THE VECTOR OF LAGRANGE MULTIPLIERS IN ITS FIRST */
/*           MC+MG ELEMENTS */
/*     MODE  IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANINGS: */
/*          MODE=1: SUCCESSFUL COMPUTATION */
/*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N<1) */
/*               3: ITERATION COUNT EXCEEDED BY NNLS */
/*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
/*               5: MATRIX E IS NOT OF FULL RANK */
/*               6: MATRIX C IS NOT OF FULL RANK */
/*               7: RANK DEFECT IN HFTI */
/*     18.5.1981, DIETER KRAFT, DFVLR OBERPFAFFENHOFEN */
/*     20.3.1987, DIETER KRAFT, DFVLR OBERPFAFFENHOFEN */
    /* Parameter adjustments */
    --d__;
    --f;
    --h__;
    --x;
    g_dim1 = *lg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    e_dim1 = *le;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    c_dim1 = *lc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --w;
    --jw;

    /* Function Body */
    *mode = 1;
    
    if( *mc > *n ){
      *mode = 2;
      return;
    }
    
    l = *n - *mc;
    mc1 = *mc + 1;
    iw = (l + 1) * (*mg + 2) + (*mg << 1) + *mc;
    ie = iw + *mc + 1;
    if__ = ie + *me * l;
    ig = if__ + *me;

    /*  TRIANGULARIZE C AND APPLY FACTORS TO E AND G */
    for( i__ = 1; i__ <= *mc; ++i__ ){
	j = MIN2( i__ + 1, *lc );
	i__2 = i__ + 1;
	i__3 = *mc - i__;
	h12_( &c__1,
	      &i__, &i__2, n,
	      &c__[i__ + c_dim1], lc, &w[iw + i__], 
	      &c__[j + c_dim1], lc, &c__1, &i__3 );
	h12_( &c__2,
	      &i__, &i__2, n,
	      &c__[i__ + c_dim1], lc, &w[iw + i__], 
	      &e[e_offset], le, &c__1, me );
	h12_( &c__2,
	      &i__, &i__2, n,
	      &c__[i__ + c_dim1], lc, &w[iw + i__], 
	      &g[g_offset], lg, &c__1, mg );
    }
    
    /*  SOLVE C*X=D AND MODIFY F */
    for( i__ = 1; i__ <= *mc; ++i__ ){
	if( fabs(c__[i__ + i__ * c_dim1]) < epmach ){
	  *mode = 6;
	  return ;
	}
	i__1 = i__ - 1;
	x[i__] = (d__[i__] - ddot_sl__(&i__1, &c__[i__ + c_dim1], *lc, &x[1], 1)) 
	         / c__[i__ + i__ * c_dim1];
    }

    w[mc1] = 0.0;
    i__2 = *mg; /* BUGFIX for *mc == *n: changed from *mg - *mc, SGJ 2010 */
    dcopy___(&i__2, &w[mc1], 0, &w[mc1], 1);
    if( *mc == *n ){ goto L50; }
    
    for( i__ = 1; i__ <= *me; ++i__ ){
	w[if__ - 1 + i__] = f[i__] - ddot_sl__(mc, &e[i__ + e_dim1], *le, &x[1], 1);
    }

    /*  STORE TRANSFORMED E & G */
    for( i__ = 1; i__ <= *me; ++i__ ){
	dcopy___(&l, &e[i__ + mc1 * e_dim1], *le, &w[ie - 1 + i__], *me);
    }
    for( i__ = 1; i__ <= *mg; ++i__ ){
	dcopy___(&l, &g[i__ + mc1 * g_dim1], *lg, &w[ig - 1 + i__], *mg);
    }
    if( *mg > 0 ){
      /*  MODIFY H AND SOLVE INEQUALITY CONSTRAINED LS PROBLEM */
      for( i__ = 1; i__ <= *mg; ++i__ ){
	h__[i__] -= ddot_sl__(mc, &g[i__ + g_dim1], *lg, &x[1], 1);
      }
      lsi_(&w[ie], &w[if__],
	   &w[ig], &h__[1],
	   me, me, mg, mg,
	   &l, &x[mc1], xnrm,
	   &w[mc1], &jw[1], mode);
      if( *mc == 0 ){ return; } // 等式制約がなければこのまま return.
      t = dnrm2___(mc, &x[1], 1);
      *xnrm = sqrt( (*xnrm)*(*xnrm) + t*t );
      if( *mode != 1 ){ return; }
    }else {
      /*  SOLVE LS WITHOUT INEQUALITY CONSTRAINTS */
      k = MAX2( *le, *n );
      t = sqrt(epmach);
      hfti_( &w[ie], me,
	     me, &l,
	     &w[if__], &k, &c__1,
	     &t, &krank, xnrm,
	     &w[1], &w[l + 1], &jw[1]);
      dcopy___(&l, &w[if__], 1, &x[mc1], 1);
      if( krank != l ){
	*mode = 7;
	return;
      }
    }

    /*  SOLUTION OF ORIGINAL PROBLEM AND LAGRANGE MULTIPLIERS */
L50:
    for( i__ = 1; i__ <= *me; ++i__ ){
	f[i__] = ddot_sl__(n, &e[i__ + e_dim1], *le, &x[1], 1) - f[i__];
    }
    for( i__ = 1; i__ <= *mc; ++i__ ){
	d__[i__] = ddot_sl__(me, &e[i__ * e_dim1 + 1], 1, &f[1], 1) 
	           - ddot_sl__(mg, &g[i__ * g_dim1 + 1], 1, &w[mc1], 1);
    }
    for( i__ = *mc; i__ >= 1; --i__ ){
	i__2 = i__ + 1;
	h12_(&c__2,
	     &i__, &i__2, n,
	     &c__[i__ + c_dim1], lc, &w[iw + i__], 
	     &x[1], &c__1, &c__1, &c__1);
    }
    for( i__ = *mc; i__ >= 1; --i__ ){
	j = MIN2( i__ + 1, *lc );
	i__2 = *mc - i__;
	w[i__] = (d__[i__] - ddot_sl__(&i__2, &c__[j + i__ * c_dim1], 1, &w[j], 1))
	         / c__[i__ + i__ * c_dim1];
    }
    
    /*  END OF SUBROUTINE LSEI */
    return;
} /* lsei_ */


// m  : 制約式の個数
// meq: 等号制約の制約式の個数.
// la : ベクトル c__ と a に関しては都合により 1 以上の長さをとらせる. そのため la = Max(m,1) として設定されて呼び出される.
// n  : 変数の個数
// nl : l の要素数. 
// la : a の縦の長さ. 通常は m だが m = 0 の場合は 1.
//  l : へシアンを LDL 分解した際の L と D を保持する. len = nl
// g  : 元の問題の目的関数の x における 1 階微分. len = n
// a  : 制約式をあらわす行列.関数内に書いてある document のとおり. len = m*n.
// a は以下のように列 wise に保持される。
// m = n = 4 の場合.
// | 1 5  9 13 |
// | 2 6 10 14 |
// | 3 7 11 15 |
// | 4 8 12 16 |
//
//   b: 制約式の右辺の値. 関数内に書いてある document のとおり. len = m.
// xl,xu : 変数 x の上下限. len = n.
//     x : この lsq_ の解. len = n
//     y : この lsq_ の解 x におけるラグランジュ乗数. len = m+n+n ( 制約式と変数の上下限の分.)
//  w,jw : lsei_ 内で使われるワークスペース.
//  mode : この関数からの状態の返り値と lsei_ の戻り値の保持に使用.
static void lsq_(int *m, int *meq, int *n, int *nl, int *la,
		 double *l, double *g, double *a, double *b, 
		 const double *xl, const double *xu, 
		 double *x, double *y, 
		 double *w, int *jw, int *mode)
{
    /* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, i1, i2, i3, i4, m1, n1, n2, n3, ic, id, ie, if__, ig, ih, il,
	     im, ip, iu, iw;
    double diag;
    int mineq;
    double xnorm;

    // 元の目的関数の 2階微分 H = LDL^t と 1 階微分 g に対して (1/2)s^t*H*s + g^t*s を
    // || D^(1/2)*L^t*s + D^(-1/2)*L^(-1)*g ||^2 + ... のように平方の形にして、
    // 目的関数が ||E*X - F|| の形のケースに持ち込んでいる.
    
/*   MINIMIZE with respect to X */
/*             ||E*X - F|| */
/*                                      1/2  T */
/*   WITH UPPER TRIANGULAR MATRIX E = +D   *L , */
/*                                      -1/2  -1 */
/*                     AND VECTOR F = -D    *L  *G, */
/*  WHERE THE UNIT LOWER TRIDIANGULAR MATRIX L IS STORED COLUMNWISE */
/*  DENSE IN THE N*(N+1)/2 ARRAY L WITH VECTOR D STORED IN ITS */
/*  'DIAGONAL' THUS SUBSTITUTING THE ONE-ELEMENTS OF L */
/*   SUBJECT TO */
/*             A(J)*X - B(J) = 0 ,         J=1,...,MEQ, */
/*             A(J)*X - B(J) >=0,          J=MEQ+1,...,M, */
/*             XL(I) <= X(I) <= XU(I),     I=1,...,N, */
/*     ON ENTRY, THE USER HAS TO PROVIDE THE ARRAYS L, G, A, B, XL, XU. */
/*     WITH DIMENSIONS: L(N*(N+1)/2), G(N), A(LA,N), B(M), XL(N), XU(N) */
/*     THE WORKING ARRAY W MUST HAVE AT LEAST THE FOLLOWING DIMENSION: */
/*     DIM(W) =        (3*N+M)*(N+1)                        for LSQ */
/*                    +(N-MEQ+1)*(MINEQ+2) + 2*MINEQ        for LSI */
/*                    +(N+MINEQ)*(N-MEQ) + 2*MEQ + N        for LSEI */
/*                      with MINEQ = M - MEQ + 2*N */
/*     ON RETURN, NO ARRAY WILL BE CHANGED BY THE SUBROUTINE. */
/*     X     STORES THE N-DIMENSIONAL SOLUTION VECTOR */
/*     Y     STORES THE VECTOR OF LAGRANGE MULTIPLIERS OF DIMENSION */
/*           M+N+N (CONSTRAINTS+LOWER+UPPER BOUNDS) */
/*     MODE  IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANINGS: */
/*          MODE=1: SUCCESSFUL COMPUTATION */
/*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N<1) */
/*               3: ITERATION COUNT EXCEEDED BY NNLS */
/*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
/*               5: MATRIX E IS NOT OF FULL RANK */
/*               6: MATRIX C IS NOT OF FULL RANK */ // 等式制約の行列が full rank でない場合.
/*               7: RANK DEFECT IN HFTI */
/*     coded            Dieter Kraft, april 1987 */
/*     revised                        march 1989 */
    /* Parameter adjustments */
    --y;
    --x;
    --xu;
    --xl;
    --g;
    --l;
    --b;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --jw;

    /* Function Body */
    n1 = *n + 1;
    mineq = *m - *meq; // 不等式制約の個数.
    m1 = mineq + *n + *n; // 変数の上下限も含めた不等式制約の個数
    /*  determine whether to solve problem */
    /*  with inconsistent linerarization (n2=1) */
    /*  or not (n2=0) */
    n2 = n1 * *n / 2 + 1; // ((n+1)*n)/2 + 1
    if( n2 == *nl ){
	n2 = 0;
    } else {
	n2 = 1;
    }
    // n2 はこの時点では inconsistent かどうかのフラグ.
    n3 = *n - n2; // n3 は通常は n と同一とみなしてよい.
    
    /*  RECOVER MATRIX E AND VECTOR F FROM L AND G */
    i2 = 1; // l において,その列の対角成分のインデックスをあらわす。
    i3 = 1; // w 上を動く
    i4 = 1; // 
    ie = 1;
    if__ = *n * *n + 1; // n*(n+1)
    for (i__ = 1; i__ <= n3; ++i__) { // l の列を iterate .
      i1 = n1 - i__; // その列に含まれる要素の個数.
      diag = sqrt(l[i2]); // Ｄ^(1/2) のその列の要素.

      // 行列 E の設定
      w[i3] = 0.0;
      dcopy___(&i1, &w[i3], 0, &w[i3], 1);
      i__2 = i1 - n2;
      dcopy___(&i__2, &l[i2], 1, &w[i3], *n); // w には L^T をコピーするため, インデックスが n とびになってることに注意.
      dscal_sl__(&i__2, &diag, &w[i3], *n);
      w[i3] = diag;

      // ベクトル F の設定
      i__2 = i__ - 1;
      w[if__ - 1 + i__] = (g[i__] - ddot_sl__(&i__2, &w[i4], 1, &w[if__], 1)) / diag;
      i2 = i2 + i1 - n2;
      i3 += n1;
      i4 += *n;
    }
    if (n2 == 1) {
	w[i3] = l[*nl];
	w[i4] = 0.0;
	dcopy___(&n3, &w[i4], 0, &w[i4], 1);
	w[if__ - 1 + *n] = 0.0;
    }
    d__1 = -one;
    dscal_sl__(n, &d__1, &w[if__], 1);
    ic = if__ + *n;
    id = ic + *meq * *n;
    if (*meq > 0) {
      /*  RECOVER MATRIX C FROM UPPER PART OF A */
	for (i__ = 1; i__ <= *meq; ++i__) {
	    dcopy___(n, &a[i__ + a_dim1], *la, &w[ic - 1 + i__], *meq);
	}
	/*  RECOVER VECTOR D FROM UPPER PART OF B */
	dcopy___(meq, &b[1], 1, &w[id], 1);
	d__1 = -one;
	dscal_sl__(meq, &d__1, &w[id], 1);
    }
    ig = id + *meq;
    if( mineq > 0 ){
      /*  RECOVER MATRIX G FROM LOWER PART OF A */
	for (i__ = 1; i__ <= mineq; ++i__) {
	    dcopy___(n, &a[*meq + i__ + a_dim1], *la, &w[ig - 1 + i__], m1);
	}
    }
    /*  AUGMENT MATRIX G BY +I AND -I */
    ip = ig + mineq;
    for (i__ = 1; i__ <= *n; ++i__) {
	w[ip - 1 + i__] = 0.0;
	dcopy___(n, &w[ip - 1 + i__], 0, &w[ip - 1 + i__], m1);
    }

    i__1 = m1 + 1;/*  注意. ここでは i__1 は i__ の長さではない! */
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
      if (!nlopt_isinf(xl[i__])){ w[(ip - i__1) + i__ * i__1] = +1.0; }
    /* Old code: w[ip] = one; dcopy___(n, &w[ip], 0, &w[ip], i__1); */
    im = ip + *n;
    for (i__ = 1; i__ <= *n; ++i__) {
	w[im - 1 + i__] = 0.0;
	dcopy___(n, &w[im - 1 + i__], 0, &w[im - 1 + i__], m1);
    }
    i__1 = m1 + 1; /*  注意. ここでは i__1 は i__ の長さではない! */
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
      if (!nlopt_isinf(xu[i__])){ w[(im - i__1) + i__ * i__1] = -1.0; }
    /* Old code: w[im] = -one;  dcopy___(n, &w[im], 0, &w[im], i__1); */
    ih = ig + m1 * (*n);
    if (mineq > 0) {
      /*  RECOVER H FROM LOWER PART OF B */
	dcopy___(&mineq, &b[*meq + 1], 1, &w[ih], 1);
	d__1 = -one;
	dscal_sl__(&mineq, &d__1, &w[ih], 1);
    }
    /*  AUGMENT VECTOR H BY XL AND XU */
    il = ih + mineq;
    iu = il + *n;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__) {
	 w[(il-1) + i__] = nlopt_isinf(xl[i__]) ? 0 : xl[i__];
	 w[(iu-1) + i__] = nlopt_isinf(xu[i__]) ? 0 : -xu[i__];
    }
    /* Old code: dcopy___(n, &xl[1], 1, &w[il], 1);
                 dcopy___(n, &xu[1], 1, &w[iu], 1);
		 d__1 = -one; dscal_sl__(n, &d__1, &w[iu], 1); */
    iw = iu + *n;
    i__1 = MAX2( 1, *meq );
    // w[ic:] = a , w[id:] = b
    // w[ie:] = e, w[if__:] = f
    // w[ig:] = g, w[ih:] = h
    // 
    lsei_(&w[ic], &w[id], 
	  &w[ie], &w[if__], 
	  &w[ig], &w[ih], 
	  &i__1, meq, 
	  n, n, 
	  &m1, &m1, 
	  n, 
	  &x[1], &xnorm, 
	  &w[iw], &jw[1], mode);
    if (*mode == 1) {
      /*   restore Lagrange multipliers */
	dcopy___(m, &w[iw], 1, &y[1], 1);
	dcopy___(&n3, &w[iw + *m], 1, &y[*m + 1], 1);
	dcopy___(&n3, &w[iw + *m + *n], 1, &y[*m + n3 + 1], 1);

	/* SGJ, 2010: make sure bound constraints are satisfied, since
	   roundoff error sometimes causes slight violations and
	   NLopt guarantees that bounds are strictly obeyed */
	// 丸め誤差で変数の上下限が破られている可能性があるので補正する. 
	for (i__ = 1; i__ <= *n; ++i__) {
	     if (x[i__] < xl[i__]) x[i__] = xl[i__];
	     else if (x[i__] > xu[i__]) x[i__] = xu[i__];
	}
    }
    /*   END OF SUBROUTINE LSQ */
} /* lsq_ */


// a = LDL' となっているときに
// a~ = a + sigma*z*z^t に対して, LDL^t 分解を導き、a に保持される。
// 理論は「On the Modification of LDL^t Factorizations」 By R. Fletcher and M. J. D. Powell
// による. p1075 にある composite t-method をそのまま組んだもの.
//
// a は n*n 行列をあらわす。
// a は以下のように column wize に LDL^t を保持.
// n = 4 のとき
// a は以下のように要素を保持.
// | 1        |
// | 2 5      |
// | 3 6 8    |
// | 4 7 9 10 |
// なので n=4 のときは a は長さ 10 (4*(4+1)/2 = 10) の配列.
// a は入力でもあり更新結果も入ってくる。
// z は長さ n のベクトル.　計算中、汚されるので注意.
// w は長さ n のベクトル. sigma < 0 の場合に計算中で使われる。
// sigma : スカラー値.
static void ldl_(int *n, double *a, double *z__, 
		 const double *sigma, double *w)
{
    /* Initialized data */

    const double one = 1.;
    const double four = 4.;
    const double epmach = 2.22e-16;

    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    int i__, j;
    double t, u, v;
    int ij;
    double tp, beta, gamma_, alpha, delta;

    
/*   LDL     LDL' - RANK-ONE - UPDATE */
/*   PURPOSE: */
/*           UPDATES THE LDL' FACTORS OF MATRIX A BY RANK-ONE MATRIX */
/*           SIGMA*Z*Z' */
/*   INPUT ARGUMENTS: (* MEANS PARAMETERS ARE CHANGED DURING EXECUTION) */
/*     N     : ORDER OF THE COEFFICIENT MATRIX A */
/*   * A     : POSITIVE DEFINITE MATRIX OF DIMENSION N; */
/*             ONLY THE LOWER TRIANGLE IS USED AND IS STORED COLUMN BY */
/*             COLUMN AS ONE DIMENSIONAL ARRAY OF DIMENSION N*(N+1)/2. */
/*   * Z     : VECTOR OF DIMENSION N OF UPDATING ELEMENTS */
/*     SIGMA : SCALAR FACTOR BY WHICH THE MODIFYING DYADE Z*Z' IS */
/*             MULTIPLIED */
/*   OUTPUT ARGUMENTS: */
/*     A     : UPDATED LDL' FACTORS */
/*   WORKING ARRAY: */
/*     W     : VECTOR OP DIMENSION N (USED ONLY IF SIGMA .LT. ZERO) */
/*   METHOD: */
/*     THAT OF FLETCHER AND POWELL AS DESCRIBED IN : */
/*     FLETCHER,R.,(1974) ON THE MODIFICATION OF LDL' FACTORIZATION. */
/*     POWELL,M.J.D.      MATH.COMPUTATION 28, 1067-1078. */
/*   IMPLEMENTED BY: */
/*     KRAFT,D., DFVLR - INSTITUT FUER DYNAMIK DER FLUGSYSTEME */
/*               D-8031  OBERPFAFFENHOFEN */
/*   STATUS: 15. JANUARY 1980 */
/*   SUBROUTINES REQUIRED: NONE */
    /* Parameter adjustments */
    --w;
    --z__;
    --a;

    /* Function Body */
    if( *sigma == 0.0 ){ return ; }
    
    ij = 1;
    t = one / *sigma; //   1 / sigma
    if( *sigma < 0.0 ){
      // sigma が負数の場合、更新式の途中に負の数がでてくることにおり、
      // 0 に近い場合に問題が生じるので、そのための対策。
      // t_i, t_i+1 ... の代わりになる数値を w に保持しておく.

      /* PREPARE NEGATIVE UPDATE */
      // w = z
      for (i__ = 1; i__ <= *n; ++i__) {
	w[i__] = z__[i__];
      }

      // L*v = z の解 v を w につめこみつつ, t の列を計算している。
      for (i__ = 1; i__ <= *n; ++i__) {
	v = w[i__];
	t += v * v / a[ij]; // t_(i+1) = t_i + v*v/d_i
	for (j = i__ + 1; j <= *n; ++j) {
	  ++ij;
	  w[j] -= v * a[ij]; // w -= v*A[*,i]
	}
	++ij;
      }
      if( t >= 0.0 ){ // 理論上は t_(n+1) < 0 なので丸め誤差が生じたと判断する。
	t = epmach / *sigma;
      }
      for (i__ = 1; i__ <= *n; ++i__) {
	j = *n + 1 - i__;
	ij -= i__; // a の対角成分をあらわすインデックス
	u = w[j];
	w[j] = t;
	t -= u * u / a[ij];
      }
    }    

    /* HERE UPDATING BEGINS */
    // t = t_i
    for (i__ = 1; i__ <= *n; ++i__) {
        v = z__[i__]; // v_i
	delta = v / a[ij]; // delta = v_i/d_i
	if( *sigma < 0.0 ){ // 前もって計算した t の列を使う。
	    tp = w[i__];
	}else /* if (*sigma > 0.0), since *sigma != 0 from above */ {
	  tp = t + delta * v; // tp = t + (v^2)/ d_i, tp は t_(i+1) をあらわす.
	}
	alpha = tp / t; // t_(i+1)/t_i
	a[ij] = alpha * a[ij]; // (d_i)^~ = d_i*t_(i+1)/t_i
	if (i__ == *n) { return; }
	
	beta = delta / tp; // beta_i = (v_i/d_i)/t_(i+1)
	if( alpha > four ){
	  gamma_ = t / tp; 
	  for (j = i__ + 1; j <= *n; ++j) {
	    ++ij;
	    u = a[ij];
	    a[ij] = gamma_ * u + beta * z__[j]; // 1 - beta*v = t_i/t_(i+1) なので同じ計算になる.
	    z__[j] -= v * u;
	  }
	}else{
	  // z^(i+1) = z^i - v_i*l_i
	  // (l_i)^~ = l_i + beta_i*z^(i+1)
	  // の更新をいっぺんに行っている。
	  for (j = i__ + 1; j <= *n; ++j) {
	    ++ij;
	    z__[j] -= v * a[ij];
	    a[ij] += beta * z__[j];
	  }
	}
	++ij;
	t = tp;
    }

    return;
    /* END OF LDL */
} /* ldl_ */

// mu で重み付けされた違反量の総和を返す。
// 1 始まりで m も含むことに注意.
static double get_weight_violate( int m, int meq, const double* mu, const double* c__ )
{
  double h = 0.0;
  for( j = 1; j <= m; ++j ){
    if( j <= meq ){
      h += mu[j] * MAX2( -c__[j], c__[j] );
    } else {
      h += mu[j] * MAX2( -c__[j], 0.0 );
    }
  }
  return h;
}

static double get_violate( int m, int meq, const double* c__ )
{
  double h = 0.0;
  for( j = 1; j <= m; ++j ){
    if( j <= meq ){
      h += MAX2( -c__[j], c__[j] );
    } else {
      h += MAX2( -c__[j], 0.0 );
    }
  }
  return h;
}

typedef struct {
    double t, f0, h1, h2, h3, h4;
    int n1, n2, n3;
    double t0, gs;
    double tol;
    int line;
    double alpha;
    int iexact;
    int incons, ireset, itermx;
    double *x0;
} slsqpb_state;

// 前回に slsqpb_ を呼び出したときのテンポラリ変数を保持するための構造.
#define SS(var) state->var = var
#define SAVE_STATE \
     SS(t); SS(f0); SS(h1); SS(h2); SS(h3); SS(h4);	\
     SS(n1); SS(n2); SS(n3); \
     SS(t0); SS(gs); \
     SS(tol); \
     SS(line); \
     SS(alpha); \
     SS(iexact); \
     SS(incons); SS(ireset); SS(itermx)

#define RS(var) var = state->var
#define RESTORE_STATE \
     RS(t); RS(f0); RS(h1); RS(h2); RS(h3); RS(h4);	\
     RS(n1); RS(n2); RS(n3); \
     RS(t0); RS(gs); \
     RS(tol); \
     RS(line); \
     RS(alpha); \
     RS(iexact); \
     RS(incons); RS(ireset); RS(itermx)



// アルゴリズムは
// 「Numerical Optimization」by Jorge Nocedal,Stephen J. Wright
// の p545 あたりの Algorithm 18.3(Line search SQP Algorithm) を実装したもの.
//
// m  : 制約式の個数
// meq: 等号制約の制約式の個数.
// la : ベクトル c__ と a に関しては都合により 1 以上の長さをとらせる. そのため la = Max(m,1) として設定されて呼び出される.
// n  : 変数の個数
//   x: 変数 x　の現在値. len = n
// xl, xu : 変数 x の上下限. len = n
//  f: 現在の目的関数の値. スカラー値.
// c__ : 現在の制約式の x における関数の値. len = m
// g   : 目的関数の x における 1 階微分. len = n
// a   : 制約式の x における 1 階微分. len = m*n.
// a は以下のように列 wise に保持される。
// m = n = 4 の場合.
// | 1 5  9 13 |
// | 2 6 10 14 |
// | 3 7 11 15 |
// | 4 8 12 16 |
//
// acc: 精度パラメータ, nlopt_slsqp からは常に *acc = 0 として呼ばれる.
// iter: x の更新回数 = while 文の回った回数.
// mode: この関数からの状態の返り値と lsq_ の戻り値の保持に使用.
//       mode = 0 となることは基本的にない。
//       (acc=0 で放り込まれるのでよほどぴったりの場合だがそれはほとんど起こらない)
// r__: 部分問題 lsq_ の解に対するラグランジュ乗数を保持する。len= m+n+n. 長さは制約式と変数の上下限制約に対するもの.
//  l: 目的関数の二階微分を近似した行列を LDL^t 分解した際に生じる L と D の成分を保持する配列.
// x0: x = x + alpha*s と更新する前の x を保持するためのバッファ配列. len = n.
// mu: L1-test を計算する際の配列. len = m. 各制約に対するラグランジュ乗数を独自手法により加算したものを保持する。
// s : lsq_ で求めたステップ方向を保持する配列. len = n
// u,v: この関数内でのワークスペース. 前半では x の上下限,後半では BFGS をアップデートする際の変数を保持.
// w,iw: lsq_ 内で使われるワークスペース.
// state: slsqpb_ 内のローカル変数の値を保持し、次に呼びだされたときにロードして再スタートできるようにするための変数
//        RESTORE_STATE, SAVE_STATE マクロにより、slsqpb_　内の変数とやり取りする。
static void slsqpb_( const int *m, const int *meq, const int *la, const int *n, 
		    double *x, const double *xl, const double *xu, double *f, 
		    double *c__, double *g, double *a,
		    double *acc, int *iter, int *mode, 
		    double *r__, double *l, double *x0, double *mu, double *s,
		    double *u, double *v,
		    double *w, int *iw, 
		    slsqpb_state *state)
{
  /* Initialized data */

  const double one = 1.;
  const double alfmin = .1;
  const double hun = 100.;
  const double ten = 10.;
  const double two = 2.;

  /* System generated locals */
  int a_dim1, a_offset, i__1, i__2;
  double d__1, d__2;

  /* Local variables */
  int i__, j, k;

  /* saved state from one call to the next;
     SGJ 2010: save/restore via state parameter, to make re-entrant. */
  double t, f0, h1, h2, h3, h4;
  int n1, n2, n3;
  double t0, gs;
  double tol;
  int line;
  double alpha;
  int iexact; // nlopt_slsqp -> slsqp から呼ばれる場合は常に *acc = 0 なので,常に iexact = 0 となる.
  int ireset, itermx;
  RESTORE_STATE;

  /*   NONLINEAR PROGRAMMING BY SOLVING SEQUENTIALLY QUADRATIC PROGRAMS */
  /*        -  L1 - LINE SEARCH,  POSITIVE DEFINITE  BFGS UPDATE  - */
  /*                      BODY SUBROUTINE FOR SLSQP */
  /*     dim(W) =         N1*(N1+1) + MEQ*(N1+1) + MINEQ*(N1+1)  for LSQ */
  /*                     +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ */
  /*                     +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1       for LSEI */
  /*                      with MINEQ = M - MEQ + 2*N1  &  N1 = N+1 */
  /* Parameter adjustments */
  --mu;
  --c__;
  --v;
  --u;
  --s;
  --x0;
  --l;
  --r__;
  a_dim1 = *la;
  a_offset = 1 + a_dim1;
  a -= a_offset;
  --g;
  --xu;
  --xl;
  --x;
  --w;
  --iw;

  /* Function Body */
  if( *mode == -1 ){ // 前回の slsqpb_ の呼び出しにおいて, L240 あたりのところの精度チェックに引っかかった場合.
    goto L260;
  } else if( *mode != 0 ){
    // *mode = -2 を想定. この関数に *mode = 1 で入ってくることはない.
    goto L220;
  }
  
  itermx = *iter;
  if (*acc >= 0.0) { 
    iexact = 0;
  } else {
    iexact = 1;
  }
  *acc = fabs(*acc);
  tol = ten * *acc;
  *iter = 0;
  ireset = 0;
  n1 = *n + 1;
  n2 = n1 * *n / 2; // ((n+1)*n)/2
  n3 = n2 + 1;

  /* initialize s[*], mu[*] = 0.0, 0.0 */
  s[1] = 0.0;
  mu[1] = 0.0;
  dcopy___(n, &s[1], 0, &s[1], 1);
  dcopy___(m, &mu[1], 0, &mu[1], 1);

  /*   RESET BFGS MATRIX */
L110:
  ++ireset;
  if( ireset > 5 ){ goto L255; }

  // initialize l = E.
  //
  // n = 4 のとき( n1 = 5 のとき)
  // l は以下のように要素を保持.
  // | 1        |
  // | 2 5      |
  // | 3 6 8    |
  // | 4 7 9 10 |
  //
  l[1] = 0.0;
  dcopy___(&n2, &l[1], 0, &l[1], 1);
  j = 1;
  for( i__ = 1; i__ <= *n; ++i__ ){
    l[j] = one;
    j = j + (n1 - i__);
  }

  /*   MAIN ITERATION : SEARCH DIRECTION, STEPLENGTH, LDL'-UPDATE */
  while(1){    
    //L130:
    ++(*iter);
    if( *iter > itermx && itermx > 0 ){ /* SGJ 2010: ignore if itermx <= 0 */
      *mode = 9; 
      break;
    }
    
    /* SEARCH DIRECTION AS SOLUTION OF QP - SUBPROBLEM */

    // s は x = x + s として解を更新するためのベクトル。
    // u,v はそれぞれ s の lower, upper であるので xl,xu から現在の x の分をひいたものの範囲にあるとする.
    dcopy___(n, &xl[1], 1, &u[1], 1); 
    dcopy___(n, &xu[1], 1, &v[1], 1); 
    d__1 = -one;
    daxpy_sl__(n, &d__1, &x[1], 1, &u[1], 1); 
    daxpy_sl__(n, &d__1, &x[1], 1, &v[1], 1); 
    h4 = one;

    // H = LDL^t と g に対して (1/2)s^t*H*s + g^t*s を
    // | D^(1/2)*L^t*s + D^(-1/2)*L^(-1)*g |^2 + ... として目的関数を設定.
    lsq_(m, meq, n, &n3, la,
	 &l[1], &g[1], &a[a_offset], &c__[1], 
	 &u[1], &v[1], 
	 &s[1], &r__[1], 
	 &w[1], &iw[1], mode);

    /*   AUGMENTED PROBLEM FOR INCONSISTENT LINEARIZATION */
    if( *mode == 6 && *n == *meq ){
      // *mode == 6 となるのは lsq_ の等式制約の rank が足りなかった場合.
      *mode = 4;
    }
    
    if( *mode == 4 ){
      // lsq_ で *mode = 4 となるのは, 不等式を満たせなかった場合.
      // n+1 個目の変数(slack変数)を付け加えて、とりあえず解が出るようにする。
      // n+1 番目の変数の値 = 1 とおけば解が出るための制約式の設定.
      for( j = 1; j <= *m; ++j ){
	if( j <= *meq ){ // 等式制約について.
	  a[j + n1 * a_dim1] = -c__[j];
	} else { // 不等式制約について.
	  a[j + n1 * a_dim1] = MAX2( -c__[j], 0.0 );
	}
      }
      // s[:] = 0.0 で初期化.
      s[1] = 0.0;
      dcopy___(n, &s[1], 0, &s[1], 1);
      h3 = 0.0;
      g[n1] = 0.0;
      l[n3] = hun;
      s[n1] = one;
      u[n1] = 0.0;
      v[n1] = one;
      
      int incons = 0; // lsq_ が infeasible だった回数.
    L150:
      lsq_(m, meq, &n1, &n3, la,
	   &l[1], &g[1], &a[a_offset], &c__[1], 
	   &u[1], &v[1],
	   &s[1], &r__[1], 
	   &w[1], &iw[1], mode);
      h4 = one - s[n1];
      if( *mode == 4 ){
	l[n3] = ten * l[n3];
	++incons;
	if( incons > 5 ){ break; } // あきらめる.
	goto L150;
      } else if ( *mode != 1 ){
	break;
      }
      
    } else if( *mode != 1 ){
      break;
    }

    /*   UPDATE MULTIPLIERS FOR L1-TEST */

    // v = g - (A^t)[*,:m] * r[:m],
    // lagrangian を x で微分したもの.
    for (i__ = 1; i__ <= *n; ++i__) {
      v[i__] = g[i__] - ddot_sl__(m, &a[i__ * a_dim1 + 1], 1, &r__[1], 1);
    }

    // 目的関数値と解を f0、x0 に退避しておく.
    f0 = *f;
    dcopy___(n, &x[1], 1, &x0[1], 1);
    
    gs = ddot_sl__(n, &g[1], 1, &s[1], 1); // 目的関数の 1 次の項の s による増減分.
    
    h1 = fabs(gs); // |g*s| + sigma_j( c_e[j]*m[j] ), c_e は等式制約のもの
    h2 = 0.0; // 等号、不等号含めた制約式の違反量の総和.(各項は違反量の絶対値)
    for( j = 1; j <= *m; ++j ){
      if( j <= *meq ){
	h2 += MAX2( -c__[j], c__[j] );
      } else {
	h2 += MAX2( -c__[j], 0.0 );
      }
      h3 = fabs( r__[j] );
      h1 += h3 * fabs( c__[j] );
      mu[j] = MAX2( h3, (mu[j] + h3)/two ); // h3 >= m[j] なら h3, h3 < m[j] なら中間の値.
    }
    
    /*   CHECK CONVERGENCE */
    if( h1 < *acc && h2 < *acc ){ // acc = 0.0 としているので、ここで終わることはない。あったら bug。
      *mode = 0;
      break;
    }
    
    h1 = get_weight_violate( *m, *meq, mu, c__ ); // 累積的な lagrange multiplier を重みとした違反量の総和.
    t0 = *f + h1; // ある種の lagrangian の値。この計算方法は少し特殊なので収束性に関しては調べる必要あり。
    h3 = gs - h1 * h4; // h4 は default では 1 だが, infeasible な場合に slack 変数を導入した場合にその slack 変数分マイナスしたもの.
    
    *mode = 8;
    if( h3 >= 0.0 ){ // メリット関数の方向微分が負でない場合、
      goto L110;
    }
    
    /*   LINE SEARCH WITH AN L1-TESTFUNCTION */
    line = 0;
    alpha = one;
    if( iexact == 1 ){
      // *acc < 0.0 の場合のみ. nlopt_slsqp では通らない.
      *mode = 9 /* will yield nlopt_failure */;
      return;
    }

    /*   INEXACT LINESEARCH */
  L190:
    ++line; 
    h3 = alpha * h3;
    
    // x = x0 + alpha*s
    dscal_sl__(n, &alpha, &s[1], 1);
    dcopy___(n, &x0[1], 1, &x[1], 1);
    daxpy_sl__(n, &one, &s[1], 1, &x[1], 1);
    
    /* SGJ 2010: ensure roundoff doesn't push us past bound constraints */
    for (i__ = 1; i__ <= *n; ++i__) {
      if (x[i__] < xl[i__]) x[i__] = xl[i__];
      else if (x[i__] > xu[i__]) x[i__] = xu[i__];
    }

    /* SGJ 2010: optimizing for the common case where the inexact line
       search succeeds in one step, use special mode = -2 here to
       eliminate a a subsequent unnecessary mode = -1 call, at the 
       expense of extra gradient evaluations when more than one inexact
       line-search step is required */
    //    *mode = line == 1 ? -2 : 1;
    *mode = line == -2;
    // lagrangian の値はそれなりに下がったけれども制約違反量、もしくは目的関数値の減少が収束しなかった場合.
    break;

    /*   CALL FUNCTIONS AT CURRENT X */
    // *mode = -2 のときはここから. *mode = -2 で始まるのは すぐ上の L190 からのところで設定された場合のみ.
  L220:
    t = *f + get_weight_violate( *m, *meq, mu, c__ ); // t = f + sigma_j( mu[j] * 制約式 j の違反量 )
    h1 = t - t0; // return 前の lagrangian との差分値
    switch( iexact ){
      case 0:
	if (nlopt_isfinite(h1)) {
	  if( h1 <= h3 / ten || line > 10 ){ // メリット関数の差分 h1 と, alpha*D_1 の比較. eta = 0.1 としている場合.
	    goto L240;
	  }else{
	    // h3 <0 なので h3/10 < h1 ならば h3/2 < h1. よって 1 より小さいステップとなる.
	    // ここの step サイズに関しては,
	    // 「Numerical Optimization」by Jorge Nocedal,Stephen J. Wright の
	    //  p57,58あたりの alpha_0 = 1 の場合の計算によっている。
	    alpha = MAX2( h3/(two * (h3 - h1)), alfmin );
	  }
	} else {
	  alpha = MAX2( alpha*.5, alfmin );
	}
	goto L190;
      case 1:
	*mode = 9 /* will yield nlopt_failure */;
	return;
    }
    
    /*   CHECK CONVERGENCE */
  L240:
    h3 = get_violate( *m, *meq, c__ ); // 違反量の総和
    if( (fabs(*f - f0) < *acc || dnrm2___(n, &s[1], 1) < *acc ) 
	 && h3 < *acc ){
      *mode = 0;
    } else {
      *mode = -1;
      // *mode == -1 で終わった場合、すぐ下の L260 からスタート.
    }
    break;

  L260:
    // *mode == -1 で終わった場合、ここから start.
    /*   CALL JACOBIAN AT CURRENT X */
    /*   UPDATE CHOLESKY-FACTORS OF HESSIAN MATRIX BY MODIFIED BFGS FORMULA */

    //
    // u = g - (A^t)[*,:m] * r[:m] - v
    // v には以前の g - (A^t)[*,:m] * r[:m] が入っていて, u は BFGS 更新の際の y を表している.
    for (i__ = 1; i__ <= *n; ++i__) {
      u[i__] = g[i__] - ddot_sl__(m, &a[i__ * a_dim1 + 1], 1, &r__[1], 1) - v[i__];
    }

    /*  v = L'*S */
    k = 0;
    for (i__ = 1; i__ <= *n; ++i__) {
      h1 = 0.0;
      ++k;
      for (j = i__ + 1; j <= *n; ++j) {
	++k;
	h1 += l[k] * s[j];
      }
      v[i__] = s[i__] + h1;
    }

    /* v = D*L'*S, : l に D の成分が入っている. */
    k = 1;
    for (i__ = 1; i__ <= *n; ++i__) {
      v[i__] = l[k] * v[i__];
      k = k + n1 - i__;
    }

    /* v = L*D*L'*S */
    for (i__ = *n; i__ >= 1; --i__) {
      h1 = 0.0;
      k = i__;
      i__1 = i__ - 1;
      for (j = 1; j <= i__1; ++j) {
	h1 += l[k] * v[j];
	k = k + *n - j;
      }
      v[i__] += h1;
    }

    // Damped BFGS Updating の式による.
    // 「Numerical Optimization」by Jorge Nocedal,Stephen J. Wright 
    // の p537 あたりにある Procedure 18.2 を参照.
    // s^t*y の値が小さい場合に、更新する B が positive definit でなくなってしまうことを恐れる。
    // そのための修正法.
    h1 = ddot_sl__(n, &s[1], 1, &u[1], 1); // (s^t)*y
    h2 = ddot_sl__(n, &s[1], 1, &v[1], 1); // (s^t)*B*s
    h3 = h2 * .2; // 0.2 * (s^t)*B*s 
    if( h1 < h3 ){ // ここは Damped BFGS Updating の式をそのまま使用
      h4 = (h2 - h3) / (h2 - h1); // (0.8*s^t*B*s)/( (s^t)*B*s - (s^t)*y  )
      h1 = h3;
      dscal_sl__(n, &h4, &u[1], 1); // y -> theta*y に更新
      d__1 = one - h4;
      daxpy_sl__(n, &d__1, &v[1], 1, &u[1], 1); // y += (1-theta)*B*s
    }
    d__1 = one / h1; //   1 / (s^t)*y 
    ldl_(n, &l[1], &u[1], &d__1, &v[1]); // B += (y*y^t)/(y^t*s), d__1 > 0 より v は汚れない.
    d__1 = -one / h2; //    1 / (s^t)*B*s
    ldl_(n, &l[1], &v[1], &d__1, &u[1]); // B += - (B_old*s*s^t*B_old)/(s^t*B_old*s).
    /*   END OF MAIN ITERATION */
    //    goto L130;
  } // end : while(1)

  /*   CHECK relaxed CONVERGENCE in case of positive directional derivative */
 L255:
  if( ( fabs(*f - f0) < tol || dnrm2___(n, &s[1], 1) < tol )
      && h3 < tol ){
    *mode = 0;
  } else {
    *mode = 8;
  }
  
  /*   END OF SLSQPB */
 L330:
  SAVE_STATE;
} /* slsqpb_ */

/* *********************************************************************** */
/*                              optimizer                               * */
/* *********************************************************************** */
static void slsqp(int *m, int *meq, int *la, int *n,
		  double *x, const double *xl, const double *xu, double *f, 
		  double *c__, double *g, double *a, 
		  double *acc, int *iter, int *mode, 
		  double *w, int *l_w__, int *jw, int *l_jw__, 
		  slsqpb_state *state)
{
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    int n1, il, im, ir, is, iu, iv, iw, ix, mineq;

/*   SLSQP       S EQUENTIAL  L EAST  SQ UARES  P ROGRAMMING */
/*            TO SOLVE GENERAL NONLINEAR OPTIMIZATION PROBLEMS */
/* *********************************************************************** */
/* *                                                                     * */
/* *                                                                     * */
/* *            A NONLINEAR PROGRAMMING METHOD WITH                      * */
/* *            QUADRATIC  PROGRAMMING  SUBPROBLEMS                      * */
/* *                                                                     * */
/* *                                                                     * */
/* *  THIS SUBROUTINE SOLVES THE GENERAL NONLINEAR PROGRAMMING PROBLEM   * */
/* *                                                                     * */
/* *            MINIMIZE    F(X)                                         * */
/* *                                                                     * */
/* *            SUBJECT TO  C (X) .EQ. 0  ,  J = 1,...,MEQ               * */
/* *                         J                                           * */
/* *                                                                     * */
/* *                        C (X) .GE. 0  ,  J = MEQ+1,...,M             * */
/* *                         J                                           * */
/* *                                                                     * */
/* *                        XL .LE. X .LE. XU , I = 1,...,N.             * */
/* *                          I      I       I                           * */
/* *                                                                     * */
/* *  THE ALGORITHM IMPLEMENTS THE METHOD OF HAN AND POWELL              * */
/* *  WITH BFGS-UPDATE OF THE B-MATRIX AND L1-TEST FUNCTION              * */
/* *  WITHIN THE STEPLENGTH ALGORITHM.                                   * */
/* *                                                                     * */
/* *    PARAMETER DESCRIPTION:                                           * */
/* *    ( * MEANS THIS PARAMETER WILL BE CHANGED DURING CALCULATION )    * */
/* *                                                                     * */
/* *    M              IS THE TOTAL NUMBER OF CONSTRAINTS, M .GE. 0      * */
/* *    MEQ            IS THE NUMBER OF EQUALITY CONSTRAINTS, MEQ .GE. 0 * */
/* *    LA             SEE A, LA .GE. MAX(M,1)                           * */
/* *    N              IS THE NUMBER OF VARIBLES, N .GE. 1               * */
/* *  * X()            X() STORES THE CURRENT ITERATE OF THE N VECTOR X  * */
/* *                   ON ENTRY X() MUST BE INITIALIZED. ON EXIT X()     * */
/* *                   STORES THE SOLUTION VECTOR X IF MODE = 0.         * */
/* *    XL()           XL() STORES AN N VECTOR OF LOWER BOUNDS XL TO X.  * */
/* *    XU()           XU() STORES AN N VECTOR OF UPPER BOUNDS XU TO X.  * */
/* *    F              IS THE VALUE OF THE OBJECTIVE FUNCTION.           * */
/* *    C()            C() STORES THE M VECTOR C OF CONSTRAINTS,         * */
/* *                   EQUALITY CONSTRAINTS (IF ANY) FIRST.              * */
/* *                   DIMENSION OF C MUST BE GREATER OR EQUAL LA,       * */
/* *                   which must be GREATER OR EQUAL MAX(1,M).          * */
/* *    G()            G() STORES THE N VECTOR G OF PARTIALS OF THE      * */
/* *                   OBJECTIVE FUNCTION; DIMENSION OF G MUST BE        * */
/* *                   GREATER OR EQUAL N+1.                             * */
/* *    A(),LA,M,N     THE LA BY N + 1 ARRAY A() STORES                  * */
/* *                   THE M BY N MATRIX A OF CONSTRAINT NORMALS.        * */
/* *                   A() HAS FIRST DIMENSIONING PARAMETER LA,          * */
/* *                   WHICH MUST BE GREATER OR EQUAL MAX(1,M).          * */
/* *    F,C,G,A        MUST ALL BE SET BY THE USER BEFORE EACH CALL.     * */
/* *  * ACC            ABS(ACC) CONTROLS THE FINAL ACCURACY.             * */
/* *                   IF ACC .LT. ZERO AN EXACT LINESEARCH IS PERFORMED,* */
/* *                   OTHERWISE AN ARMIJO-TYPE LINESEARCH IS USED.      * */
/* *  * ITER           PRESCRIBES THE MAXIMUM NUMBER OF ITERATIONS.      * */
/* *                   ON EXIT ITER INDICATES THE NUMBER OF ITERATIONS.  * */
/* *  * MODE           MODE CONTROLS CALCULATION:                        * */ // mode >= 10 以外のところは slsqpb_ で返ってきたものを、そのまま返す.
/* *                   REVERSE COMMUNICATION IS USED IN THE SENSE THAT   * */
/* *                   THE PROGRAM IS INITIALIZED BY MODE = 0; THEN IT IS* */
/* *                   TO BE CALLED REPEATEDLY BY THE USER UNTIL A RETURN* */
/* *                   WITH MODE .NE. IABS(1) TAKES PLACE.               * */
/* *                   IF MODE = -1 GRADIENTS HAVE TO BE CALCULATED,     * */
/* *                   WHILE WITH MODE = 1 FUNCTIONS HAVE TO BE CALCULATED */
/* *                   MODE MUST NOT BE CHANGED BETWEEN SUBSEQUENT CALLS * */
/* *                   OF SQP.                                           * */
/* *                   EVALUATION MODES:                                 * */
/* *        MODE = -2,-1: GRADIENT EVALUATION, (G&A)                     * */
/* *                0: ON ENTRY: INITIALIZATION, (F,G,C&A)               * */
/* *                   ON EXIT : REQUIRED ACCURACY FOR SOLUTION OBTAINED * */
/* *                1: FUNCTION EVALUATION, (F&C)                        * */
/* *                                                                     * */
/* *                   FAILURE MODES:                                    * */
/* *                2: NUMBER OF EQUALITY CONTRAINTS LARGER THAN N       * */
/* *                3: MORE THAN 3*N ITERATIONS IN LSQ SUBPROBLEM        * */
/* *                4: INEQUALITY CONSTRAINTS INCOMPATIBLE               * */
/* *                5: SINGULAR MATRIX E IN LSQ SUBPROBLEM               * */
/* *                6: SINGULAR MATRIX C IN LSQ SUBPROBLEM               * */
/* *                7: RANK-DEFICIENT EQUALITY CONSTRAINT SUBPROBLEM HFTI* */
/* *                8: POSITIVE DIRECTIONAL DERIVATIVE FOR LINESEARCH    * */
/* *                9: MORE THAN ITER ITERATIONS IN SQP                  * */
/* *             >=10: WORKING SPACE W OR JW TOO SMALL,                  * */
/* *                   W SHOULD BE ENLARGED TO L_W=MODE/1000             * */
/* *                   JW SHOULD BE ENLARGED TO L_JW=MODE-1000*L_W       * */
/* *  * W(), L_W       W() IS A ONE DIMENSIONAL WORKING SPACE,           * */
/* *                   THE LENGTH L_W OF WHICH SHOULD BE AT LEAST        * */
/* *                   (3*N1+M)*(N1+1)                        for LSQ    * */
/* *                  +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ         for LSI    * */
/* *                  +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1       for LSEI   * */
/* *                  + N1*N/2 + 2*M + 3*N + 3*N1 + 1         for SLSQPB * */
/* *                   with MINEQ = M - MEQ + 2*N1  &  N1 = N+1          * */
/* *        NOTICE:    FOR PROPER DIMENSIONING OF W IT IS RECOMMENDED TO * */
/* *                   COPY THE FOLLOWING STATEMENTS INTO THE HEAD OF    * */
/* *                   THE CALLING PROGRAM (AND REMOVE THE COMMENT C)    * */
/* ####################################################################### */
/*     INT LEN_W, LEN_JW, M, N, N1, MEQ, MINEQ */
/*     PARAMETER (M=... , MEQ=... , N=...  ) */
/*     PARAMETER (N1= N+1, MINEQ= M-MEQ+N1+N1) */
/*     PARAMETER (LEN_W= */
/*    $           (3*N1+M)*(N1+1) */
/*    $          +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ */
/*    $          +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1 */
/*    $          +(N+1)*N/2 + 2*M + 3*N + 3*N1 + 1, */
/*    $           LEN_JW=MINEQ) */
/*     DOUBLE PRECISION W(LEN_W) */
/*     INT          JW(LEN_JW) */
/* ####################################################################### */
/* *                   THE FIRST M+N+N*N1/2 ELEMENTS OF W MUST NOT BE    * */
/* *                   CHANGED BETWEEN SUBSEQUENT CALLS OF SLSQP.        * */
/* *                   ON RETURN W(1) ... W(M) CONTAIN THE MULTIPLIERS   * */
/* *                   ASSOCIATED WITH THE GENERAL CONSTRAINTS, WHILE    * */
/* *                   W(M+1) ... W(M+N(N+1)/2) STORE THE CHOLESKY FACTOR* */
/* *                   L*D*L(T) OF THE APPROXIMATE HESSIAN OF THE        * */
/* *                   LAGRANGIAN COLUMNWISE DENSE AS LOWER TRIANGULAR   * */
/* *                   UNIT MATRIX L WITH D IN ITS 'DIAGONAL' and        * */
/* *                   W(M+N(N+1)/2+N+2 ... W(M+N(N+1)/2+N+2+M+2N)       * */
/* *                   CONTAIN THE MULTIPLIERS ASSOCIATED WITH ALL       * */
/* *                   ALL CONSTRAINTS OF THE QUADRATIC PROGRAM FINDING  * */
/* *                   THE SEARCH DIRECTION TO THE SOLUTION X*           * */
/* *  * JW(), L_JW     JW() IS A ONE DIMENSIONAL INT WORKING SPACE   * */
/* *                   THE LENGTH L_JW OF WHICH SHOULD BE AT LEAST       * */
/* *                   MINEQ                                             * */
/* *                   with MINEQ = M - MEQ + 2*N1  &  N1 = N+1          * */
/* *                                                                     * */
/* *  THE USER HAS TO PROVIDE THE FOLLOWING SUBROUTINES:                 * */
/* *     LDL(N,A,Z,SIG,W) :   UPDATE OF THE LDL'-FACTORIZATION.          * */
/* *     LINMIN(A,B,F,TOL) :  LINESEARCH ALGORITHM IF EXACT = 1          * */
/* *     LSQ(M,MEQ,LA,N,NC,C,D,A,B,XL,XU,X,LAMBDA,W,....) :              * */
/* *                                                                     * */
/* *        SOLUTION OF THE QUADRATIC PROGRAM                            * */
/* *                QPSOL IS RECOMMENDED:                                * */
/* *     PE GILL, W MURRAY, MA SAUNDERS, MH WRIGHT:                      * */
/* *     USER'S GUIDE FOR SOL/QPSOL:                                     * */
/* *     A FORTRAN PACKAGE FOR QUADRATIC PROGRAMMING,                    * */
/* *     TECHNICAL REPORT SOL 83-7, JULY 1983                            * */
/* *     DEPARTMENT OF OPERATIONS RESEARCH, STANFORD UNIVERSITY          * */
/* *     STANFORD, CA 94305                                              * */
/* *     QPSOL IS THE MOST ROBUST AND EFFICIENT QP-SOLVER                * */
/* *     AS IT ALLOWS WARM STARTS WITH PROPER WORKING SETS               * */
/* *                                                                     * */
/* *     IF IT IS NOT AVAILABLE USE LSEI, A CONSTRAINT LINEAR LEAST      * */
/* *     SQUARES SOLVER IMPLEMENTED USING THE SOFTWARE HFTI, LDP, NNLS   * */
/* *     FROM C.L. LAWSON, R.J.HANSON: SOLVING LEAST SQUARES PROBLEMS,   * */
/* *     PRENTICE HALL, ENGLEWOOD CLIFFS, 1974.                          * */
/* *     LSEI COMES WITH THIS PACKAGE, together with all necessary SR's. * */
/* *                                                                     * */
/* *     TOGETHER WITH A COUPLE OF SUBROUTINES FROM BLAS LEVEL 1         * */
/* *                                                                     * */
/* *     SQP IS HEAD SUBROUTINE FOR BODY SUBROUTINE SQPBDY               * */
/* *     IN WHICH THE ALGORITHM HAS BEEN IMPLEMENTED.                    * */
/* *                                                                     * */
/* *  IMPLEMENTED BY: DIETER KRAFT, DFVLR OBERPFAFFENHOFEN               * */
/* *  as described in Dieter Kraft: A Software Package for               * */
/* *                                Sequential Quadratic Programming     * */
/* *                                DFVLR-FB 88-28, 1988                 * */
/* *  which should be referenced if the user publishes results of SLSQP  * */
/* *                                                                     * */
/* *  DATE:           APRIL - OCTOBER, 1981.                             * */
/* *  STATUS:         DECEMBER, 31-ST, 1984.                             * */
/* *  STATUS:         MARCH   , 21-ST, 1987, REVISED TO FORTAN 77        * */
/* *  STATUS:         MARCH   , 20-th, 1989, REVISED TO MS-FORTRAN       * */
/* *  STATUS:         APRIL   , 14-th, 1989, HESSE   in-line coded       * */
/* *  STATUS:         FEBRUARY, 28-th, 1991, FORTRAN/2 Version 1.04      * */
/* *                                         accepts Statement Functions * */
/* *  STATUS:         MARCH   ,  1-st, 1991, tested with SALFORD         * */
/* *                                         FTN77/386 COMPILER VERS 2.40* */
/* *                                         in protected mode           * */
/* *                                                                     * */
/* *********************************************************************** */
/* *                                                                     * */
/* *  Copyright 1991: Dieter Kraft, FHM                                  * */
/* *                                                                     * */
/* *********************************************************************** */
/*     dim(W) =         N1*(N1+1) + MEQ*(N1+1) + MINEQ*(N1+1)  for LSQ */
/*                    +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ          for LSI */
/*                    +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1        for LSEI */
/*                    + N1*N/2 + 2*M + 3*N +3*N1 + 1           for SLSQPB */
/*                      with MINEQ = M - MEQ + 2*N1  &  N1 = N+1 */
/*   CHECK LENGTH OF WORKING ARRAYS */
    /* Parameter adjustments */
    --c__;
    a_dim1 = *la; // a の縦の長さ.
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --g;
    --xu;
    --xl;
    --x;
    --w;
    --jw;

    /* Function Body */
    n1 = *n + 1;
    mineq = *m - *meq + n1 + n1;
    il = (n1 * 3 + *m) * (n1 + 1) + (n1 - *meq + 1) * (mineq + 2)
         + (mineq << 1) + (n1 + mineq) * (n1 - *meq)
         + (*meq << 1) + n1 * *n / 2 + (*m << 1)
         + *n * 3 + (n1 << 2) + 1;
	    
    i__1 = mineq, i__2 = n1 - *meq;
    im = MAX2( i__1, i__2 );
    if( *l_w__ < il || *l_jw__ < im ){
	*mode = MAX2(10,il) * 1000;
	*mode += MAX2(10,im);
	return;
    }
    
    /*   PREPARE DATA FOR CALLING SQPBDY  -  INITIAL ADDRESSES IN W */
    im = 1; // len = *la := MAX2( 1, m )
    il = im + *la; // len = ((n+1)*n)/2 + 1, 下半行列 L　を保持する配列.
    ix = il + n1 * *n / 2 + 1; // len = n, // slsqpb_ 内において更新前の x を保持する配列.
    ir = ix + *n; // len = n + n +  MAX2( 1, m ), lsq_ の y に相当する配列. 変数 x の上下限、制約式のラグランジュ乗数を保持する配列。
    is = ir + *n + *n + *la; // len = n,  lsq_ の x に相当する配列.
    iu = is + n1; //x の下限.(slsqpb_ 内では更新される.)
    iv = iu + n1; //x の上限.(slsqpb_ 内では更新される.)
    iw = iv + n1; // lsq_ におけるワークスペース.
    slsqpb_(m, meq, la, n, 
	    &x[1], &xl[1], &xu[1], f, 
	    &c__[1], &g[1], &a[a_offset],
	    acc, iter, mode, 
	    &w[ir], &w[il], &w[ix], &w[im], &w[is],
	    &w[iu], &w[iv],
	    &w[iw], &jw[1], 
	    state);
    state->x0 = &w[ix];
    return;
} /* slsqp_ */

// len_w, len_jw の設定.
// 関数の奥深くでよばれるサブルーチンの分のワークスペース分も考慮にいれている。
static void length_work(int *LEN_W, int *LEN_JW, int M, int MEQ, int N)
{
     int N1 = N+1, MINEQ = M-MEQ+N1+N1;
     *LEN_W = (3*N1+M)*(N1+1) 
	  +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ
          +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1
          +(N+1)*N/2 + 2*M + 3*N + 3*N1 + 1;
     *LEN_JW = MINEQ;
}

// n: 変数 x の長さ.
// f :目的関数を計算する object
// f_data: f で使用する付属の object
// m : 不等式制約式の group をあらわす fc の個数
// fc : 不等式制約式の group をあらわす配列. 長さは fc 
// p : 等式制約式の group をあらわす h の個数
// h : 等式制約式の group をあらわす配列.長さは p.
// lb : 変数の下限をあらわす配列. 長さは n.
// ub : 変数の上限をあらわす配列. 長さは n.
// x : 解 x が出力される。
// minf: 解 x に対する目的関数値
// stop: 計算の停止条件に間係する情報を保持する object.
nlopt_result nlopt_slsqp(unsigned n, nlopt_func f, void *f_data,
			 unsigned m, nlopt_constraint *fc, // m = len(fc) : "<="
			 unsigned p, nlopt_constraint *h, // p = len(h)   : "=="
			 const double *lb, const double *ub,
			 double *x, double *minf,
			 nlopt_stopping *stop )
{
     slsqpb_state state = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL};

     //nlopt_constraint のひとつが複数の制約からなる場合があるのでそれを考慮して数える.
     unsigned mtot = nlopt_count_constraints(m, fc); // : "<=" の個数. 
     unsigned ptot = nlopt_count_constraints(p, h); // : "==" の個数.
     
     double fcur, fprev, 
     int mpi = (int) (mtot + ptot), pi = (int) ptot,  ni = (int) n, mpi1 = mpi > 0 ? mpi : 1;
     int len_w, len_jw;
     int mode = 0, prev_mode = 0;
     double acc = 0; /* we do our own convergence tests below */
     int iter = 0; /* tell sqsqp to ignore this check, since we check evaluation counts ourselves */
     unsigned i, ii;
     nlopt_result ret = NLOPT_SUCCESS;
     // feasible_cur は解 xcur が制約を満たしていれば1, そうでなければ 0.
     // feasible は最終的に制約を満たす x が見つかっていれば 1, そうでなければ 0.
     int feasible, feasible_cur; 
     double infeasibility = HUGE_VAL, infeasibility_cur = HUGE_VAL;
     unsigned max_cdim;
     int want_grad = 1; // gradient の更新のためのワークスペースの設定が必要な場合 1, そうでなければ 0.
     
     max_cdim = MAX2( nlopt_max_constraint_dim(m, fc),
		      nlopt_max_constraint_dim(p, h) );
     length_work( &len_w, &len_jw, mpi, pi, ni );

#define U(n) ((unsigned) (n))
     double *work = (double *) malloc(sizeof(double) * (U(mpi1) * (n + 1) 
						+ U(mpi) 
						+ n+1 + n + n + max_cdim*n
						+ U(len_w))
			      + sizeof(int) * U(len_jw));
     if (!work) return NLOPT_OUT_OF_MEMORY;
     double *cgrad = work; // len = U(mpi1) * (n + 1),  各制約関数の現在の xcur における勾配の値が入る。
     double *c = cgrad + U(mpi1) * (n + 1); // len = mpi, 各制約関数の現在の xcur における値が入る。
     double *grad = c + mpi; // len = n+1, // 目的関数の勾配の値が入る。
     double *xcur = grad + n+1; // len = n, 現在のイテレーションにおける x の値が入る.
     double *xprev = xcur + n; // len = n, ひとつ前のイテレーションにおける x の値が入る.
     double *cgradtmp = xprev + n; // len = max_cdim*n, 現在の xcur における cgrad の値が入る.

     // w, jw の長さについては slsqp の奥底のコードまで見ないとなんでこの長さになってるかはわからないと思う。
     // double と int の配列は最適化計算ではよく使うため、こんな感じではじめにメモリを確保するのはよくあること。
     double *w = cgradtmp + max_cdim*n; // len = len_w, slsqp の内部で使用するアルゴリズムのためのメモリスペース.
     int *jw = (int *) (w + len_w); // len = len_jw, slsqp の内部で使用するアルゴリズムのためのメモリスペース.
     
     memcpy(xcur, x, sizeof(double) * n);
     memcpy(xprev, x, sizeof(double) * n);
     fprev = fcur = *minf = HUGE_VAL;
     feasible = feasible_cur = 0; 

     goto eval_f_and_grad; /* eval before calling slsqp the first time */

     do {
	  slsqp(&mpi, &pi, &mpi1, &ni,
		xcur, lb, ub, &fcur,
		c, grad, cgrad,
		&acc, &iter, &mode,
		w, &len_w, jw, &len_jw,
		&state );

	  switch (mode) {
	  case -1:  /* objective & gradient evaluation */
	    if( prev_mode == -2 && !want_grad ){
	      // この場合、すでに微分の再計算は済んでいるのですぐに次のステップに行ってよいということ.
	      break;
	    } /* just evaluated this point */
	  case -2:
	      eval_f_and_grad:
	      want_grad = 1;
	  case 1:{ /* don't need grad unless we don't have it yet */
	      double *newgrad = 0;
	      double *newcgrad = 0;
	      if (want_grad) {
		  newgrad = grad;
		  newcgrad = cgradtmp;
	      }
	      feasible_cur = 1; infeasibility_cur = 0;
	      fcur = f(n, xcur, newgrad, f_data); // xcur に対する f の値と勾配を計算.
	      stop->nevals++;
	      if (nlopt_stop_forced(stop)) {
		  fcur = HUGE_VAL; ret = NLOPT_FORCED_STOP; goto done;
	      }
	      if (nlopt_isfinite(fcur)) {
		  want_grad = 0;
		  ii = 0;
		  for (i = 0; i < p; ++i) { // 等式制約に関して,
		      unsigned j, k;
		      nlopt_eval_constraint(c+ii, newcgrad, h+i, n, xcur);
		      if (nlopt_stop_forced(stop)) {
			  ret = NLOPT_FORCED_STOP; goto done;
		      }
		      for (k = 0; k < h[i].m; ++k, ++ii) {
			  infeasibility_cur = MAX2(infeasibility_cur, fabs(c[ii]));
			  feasible_cur = feasible_cur && (fabs(c[ii]) <= h[i].tol[k] );
			  if (newcgrad) {
			      for (j = 0; j < n; ++ j)
				  cgrad[j*U(mpi1) + ii] = cgradtmp[k*n + j];
			  }
		      }
		  }
		  for (i = 0; i < m; ++i) {
		      unsigned j, k;
		      nlopt_eval_constraint(c+ii, newcgrad, fc+i, n, xcur);
		      if (nlopt_stop_forced(stop)) {
			  ret = NLOPT_FORCED_STOP; goto done; }
		      for (k = 0; k < fc[i].m; ++k, ++ii) {
			  infeasibility_cur = MAX2(infeasibility_cur, c[ii]);
			  feasible_cur = feasible_cur && c[ii] <= fc[i].tol[k];
			  if( newcgrad ){
			      for (j = 0; j < n; ++ j)
				cgrad[j*U(mpi1) + ii] = -cgradtmp[k*n + j]; /* slsqp sign convention */
			  }
			  c[ii] = -c[ii]; /* slsqp sign convention */
		      }
		  }
	      }
	      break;
	  }
	  case 0: /* required accuracy for solution obtained */
	    goto done;
	  // ----------------------------------------------------------------- upper success case.
   	  // ----------------------------------------------------------------- lower ailure case
	    
	  case 8: /* positive directional derivative for linesearch */
		  /* relaxed convergence check for a feasible_cur point,
		     as in the SLSQP code (except xtol as well as ftol) */
	    // 二次近似した問題が何らかの意味で解けない場合もここにくる。
	    // 制約式が満たせない, 目的関数の二次近似がランク落ちしているなど.
	    ret = NLOPT_ROUNDOFF_LIMITED; /* usually why deriv>0 */
	    if( feasible_cur ){
	      double save_ftol_rel = stop->ftol_rel;
	      double save_xtol_rel = stop->xtol_rel;
	      double save_ftol_abs = stop->ftol_abs;
	      stop->ftol_rel *= 10;
	      stop->ftol_abs *= 10;
	      stop->xtol_rel *= 10;
	      if( nlopt_stop_ftol(stop, fcur, state.f0) ){
		ret = NLOPT_FTOL_REACHED;
	      }else if (nlopt_stop_x(stop, xcur, state.x0)){
		ret = NLOPT_XTOL_REACHED;
	      }
	      stop->ftol_rel = save_ftol_rel;
	      stop->ftol_abs = save_ftol_abs;
	      stop->xtol_rel = save_xtol_rel;
	    }
	    goto done;
	  case 5: /* singular matrix E in LSQ subproblem */
	  case 6: /* singular matrix C in LSQ subproblem */
	  case 7: /* rank-deficient equality constraint subproblem HFTI */
	    ret = NLOPT_ROUNDOFF_LIMITED;
	    goto done;
	  case 4: /* inequality constraints incompatible */
	  case 3: /* more than 3*n iterations in LSQ subproblem */
	  case 9: /* more than iter iterations in SQP */
	    nlopt_stop_msg(stop, "bug: more than iter SQP iterations");
	    ret = NLOPT_FAILURE;
	    goto done;
	  case 2: /* number of equality constraints > n */
	  default: /* >= 10: working space w or jw too small */
	    nlopt_stop_msg(stop, "bug: workspace is too small");
	    ret = NLOPT_INVALID_ARGS;
	    goto done;
	  } // 	end of  switch (mode)
	  prev_mode = mode;

	  /* update best point so far */
	  if( nlopt_isfinite(fcur)
	      && ( ( fcur < *minf && (feasible_cur || !feasible) )
		   || (!feasible && infeasibility_cur < infeasibility)) ){
	       *minf = fcur;
	       feasible = feasible_cur;
	       infeasibility = infeasibility_cur;
	       memcpy(x, xcur, sizeof(double)*n);
	  }

	  /* note: mode == -1 corresponds to the completion of a line search,
	     and is the only time we should check convergence (as in original slsqp code) */
	  if( mode == -1 ){
	    if( !nlopt_isinf(fprev) ){
	      if( nlopt_stop_ftol(stop, fcur, fprev) ){
		ret = NLOPT_FTOL_REACHED;
	      }else if( nlopt_stop_x(stop, xcur, xprev) ){
		ret = NLOPT_XTOL_REACHED;
	      }
	    }
	    fprev = fcur;
	    memcpy(xprev, xcur, sizeof(double)*n);
	  }

	  /* do some additional termination tests */
	  if( nlopt_stop_evals(stop) ){ ret = NLOPT_MAXEVAL_REACHED; }
	  else if( nlopt_stop_time(stop) ){ ret = NLOPT_MAXTIME_REACHED; }
	  else if( feasible && *minf < stop->minf_max){ ret = NLOPT_MINF_MAX_REACHED; }
     } while( ret == NLOPT_SUCCESS );

done:
     if (nlopt_isinf(*minf)) { /* didn't find any feasible points, just return last point evaluated */
	  if (nlopt_isinf(fcur)) { /* invalid cur. point, use previous pt. */
	       *minf = fprev;
	       memcpy(x, xprev, sizeof(double)*n);
	  }else {
	       *minf = fcur;
	       memcpy(x, xcur, sizeof(double)*n);
	  }
     }

     free(work);
     return ret;
}
