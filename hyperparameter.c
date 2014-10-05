#include "hyperparameter.h"

/*
  (p 585 in escobar and west)

  function of a,b,k,n,\alpha

  1. sample \eta ~ B(\alpha + 1, n)
  where n = the number of documents allocated to me

  2. sample c ~ Bern(\pi)
  where \pi/(1-\pi) = a+k-1 / n(b-log(\eta))

  3. if c = 0 then
        sample \alpha' ~ G(a + k, b - log(\eta))
     else
        sample \alpha' ~ G(a + k - 1, b - log(\eta))

*/

double gibbs_sample_DP_scaling(double alpha, // current alpha
                               double shape, // Gamma shape parameter
                               double scale, // Gamma scale parameter
                               int k,        // number of components
                               int n)        // number of data points
{
    printf("alpha=%g\nshape=%g\nscale=%g\nk=%d\nn=%d\n",
           alpha, shape, scale, k, n);

    double eta = rbeta(alpha + 1, (double) n);
    double pi = shape + k - 1;
    double rate = 1.0/scale - log(eta);
    pi = pi / (pi + rate * n);
    int c = rbernoulli(pi);

    double alpha_new = 0;
    if (c == 0)
    {
        alpha_new = rgamma(shape + k - 1, 1.0/rate);
    }
    else
    {
        alpha_new = rgamma(shape + k, 1.0/rate);
    }

    printf("-----\nnew alpha=%g\n-----\n", alpha_new);


    return(alpha_new);
}

