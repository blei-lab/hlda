#ifndef UTILSH
#define UTILSH

#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_permutation.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdarg.h>

time_t TIME;
char line1[80];

void outlog(char* fmt, ...);

typedef struct int_vector
{
    int size;
    int* val;
} int_vector;

int_vector* new_int_vector(int size);
void delete_int_vector(int_vector* iv);

void ivappend(int_vector* v, int val);

static inline int ivget(int_vector* v, int n)
{ return(v->val[n]); };

static inline void ivset(int_vector* v, int n, int val)
{ v->val[n] = val; };

static inline void vset(gsl_vector* v, int i, double x)
{ gsl_vector_set(v, i, x); };

static inline double vget(const gsl_vector* v, int i)
{ return(gsl_vector_get(v, i)); };

static inline void vinc(gsl_vector* v, int i, double x)
{ vset(v, i, vget(v, i) + x); };

static inline double log_sum(double log_a, double log_b)
{
  if (log_a < log_b)
      return(log_b+log(1 + exp(log_a-log_b)));
  else
      return(log_a+log(1 + exp(log_b-log_a)));
};

static inline double lgam(double x)
{ return(gsl_sf_lngamma(x)); }

double runif();

double rgauss(double mean, double stdev);

int sample_from_log(gsl_vector* log_prob);

void init_random_number_generator();

void vct_fscanf(const char* filename, gsl_vector* v);

void iv_permute(int_vector* iv);
void iv_permute_from_perm(int_vector* iv, gsl_permutation* p);
int_vector* iv_copy(int_vector* iv);

double sum(gsl_vector* v);

void print_vector(gsl_vector* v);

gsl_permutation* rpermutation(int size);

void write_vect(gsl_vector* vect, char* name, FILE* file);

gsl_vector* read_vect(char* name, int size, FILE* file);

int read_int(char* name, FILE* file);

void write_int(int x, char* name, FILE* file);

void write_double(double x, char* name, FILE* file);

double read_double(char* name, FILE* file);

int directory_exist(const char *dname);

void make_directory(char* name);

int rbernoulli(double p);

double rbeta(double a,
             double b);

double rgamma(double shape,
              double scale);


double log_dgamma(double x,
              double shape,
              double scale);

void resize(gsl_vector * vec, size_t newsize);

#endif
