#include "utils.h"
gsl_rng* RANDNUMGEN = NULL;

/*
 * initialize the GSL random number generator
 *
 */

void init_random_number_generator()
{
    if (RANDNUMGEN != NULL) return;

    RANDNUMGEN = gsl_rng_alloc(gsl_rng_taus);
    long t1;
    (void) time(&t1);
    // !!! DEBUG
    // t1 = 1147530551;
    // t1 = 1201962438;
    outlog("random seed = %ld\n", t1);
    gsl_rng_set (RANDNUMGEN, t1);
}


/*
 * check if a directory exists
 *
 */

int directory_exist(const char *dname)
{
	struct stat st;
	int ret;

	if (stat(dname,&st) != 0)
        {
            return 0;
	}

	ret = S_ISDIR(st.st_mode);

	if(!ret)
        {
            errno = ENOTDIR;
        }

	return ret;
}


/*
 * make directory
 *
 */

void make_directory(char* name)
{
    mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}


/*
 * sample from unnnormalized log probabilities
 *
 */

int sample_from_log(gsl_vector* log_prob)
{
    double logsum = vget(log_prob, 0);
    int i;
    for (i = 1; i < log_prob->size; i++)
    {
        logsum = log_sum(logsum, vget(log_prob, i));
    }

    double x = runif();

    double rolling_sum = exp(vget(log_prob, 0) - logsum);
    int result = 0;
    while (x >= rolling_sum)
    {
        result++;
        rolling_sum += exp(vget(log_prob, result) - logsum);
    }

    return(result);
}


/*
 * new integer vector
 *
 */

int_vector* new_int_vector(int size)
{
    int_vector* iv = malloc(sizeof(int_vector));

    iv->size = size;
    iv->val = malloc(sizeof(int) * iv->size);
    int i;
    for (i = 0; i < iv->size; i++)
        ivset(iv, i, 0);

    return(iv);
}

void delete_int_vector(int_vector* iv)
{
    free(iv->val);
    free(iv);
}


/*
 * append value to an integer vector
 *
 */

void ivappend(int_vector* iv, int val)
{
    iv->size += 1;
    iv->val = (int*) realloc(iv->val, sizeof(int) * (iv->size));
    iv->val[iv->size - 1] = val;
}


/*
 * read a vector from file
 *
 */

void vct_fscanf(const char* filename, gsl_vector* v)
{
    outlog("reading %ld vector from %s", v->size, filename);
    FILE* fileptr;
    fileptr = fopen(filename, "r");
    gsl_vector_fscanf(fileptr, v);
    fclose(fileptr);
}

/*
 * copy an integer vector
 *
 */

int_vector* iv_copy(int_vector* iv)
{
    int_vector* ivc = malloc(sizeof(int_vector));
    ivc->size = iv->size;
    ivc->val = malloc(sizeof(int) * ivc->size);

    int i;
    for (i = 0; i < ivc->size; i++)
        ivc->val[i] = iv->val[i];

    return(ivc);
}


/*
 * permute an integer vector
 *
 */

void iv_permute(int_vector* iv)
{
    gsl_ran_shuffle(RANDNUMGEN,
                    iv->val,
                    iv->size,
                    sizeof(int));
}

void iv_permute_from_perm(int_vector* iv, gsl_permutation* p)
{
    assert(iv->size == p->size);
    int_vector* ivc = iv_copy(iv);
    int i;
    for (i = 0; i < p->size; i++)
    {
        ivset(iv, i, ivget(ivc, p->data[i]));
    }
    delete_int_vector(ivc);
}


/*
 * get a random permutation
 *
 */

gsl_permutation* rpermutation(int size)
{
    gsl_permutation* perm = gsl_permutation_calloc(size);
    gsl_ran_shuffle(RANDNUMGEN,
                    perm->data, size, sizeof(size_t));
    return(perm);
}


/*
 * draw random numbers
 *
 */

double runif()
{
    return(gsl_rng_uniform(RANDNUMGEN));
}


double rgauss(double mean,
              double stdev)
{
    double v =
        gsl_ran_gaussian_ratio_method(RANDNUMGEN, stdev) + mean;
    return(v);
}

double rgamma(double shape,
              double scale)
{
    double v = gsl_ran_gamma(RANDNUMGEN, shape, scale);
    return(v);
}

double log_dgamma(double x,
                  double shape,
                  double scale)
{
    // f(x)= - a * log(s) + log_gamma(a) + (a-1) * log(x) - x/s

    double v = - shape*log(scale)+lgam(shape)+(shape-1)*log(x)-x/scale;
    return(v);
}


double rbeta(double a,
             double b)
{
    double v = gsl_ran_beta(RANDNUMGEN, a, b);
    return(v);
}

int rbernoulli(double p)
{
    int v = gsl_ran_bernoulli(RANDNUMGEN, p);
    return(v);
}

/*
 * sum a vector
 *
 */

double sum(gsl_vector* v)
{
    int i;
    double sum = 0;
    for (i = 0; i < v->size; i++)
        sum += vget(v, i);

    return(sum);
}


/*
 * print a vector
 *
 */

void print_vector(gsl_vector* v)
{
    int i;
    for (i = 0; i < v->size; i++)
    {
        printf("     [%d] %5.2e\n", i, vget(v, i));
    }
}


/*
 * read/write a vector and name to a file
 *
 */

void write_vect(gsl_vector* vect, char* name, FILE* file)
{
    int i;
    fprintf(file, "%-10s", name);
    for (i = 0; i < vect->size; i++)
    {
        fprintf(file, " %17.14e", vget(vect, i));
    }
    fprintf(file, "\n");
}


gsl_vector* read_vect(char* name, int size, FILE* file)
{
    outlog("reading %d-vector %s", size, name);

    gsl_vector* ret = gsl_vector_alloc(size);
    int i;
    float v;

    fscanf(file, name);
    for (i = 0; i < ret->size; i++)
    {
        fscanf(file, " %f", &v);
        vset(ret, i, v);
    }
    fscanf(file, "\n");

    print_vector(ret);

    return(ret);
}

/*
 * read a named integer from a file
 *
 */

int read_int(char* name, FILE* file)
{
    outlog("reading integer %s", name);
    int ret;
    fscanf(file, name);
    fscanf(file, "%d\n", &ret);
    outlog("read %d\n", ret);
    return(ret);
}

void write_int(int x, char* name, FILE* file)
{
    fprintf(file, name);
    fprintf(file, " %d\n", x);
}


void write_double(double x, char* name, FILE* file)
{
    fprintf(file, name);
    fprintf(file, " %17.14e\n", x);
}

double read_double(char* name, FILE* file)
{
    double ret;
    fscanf(file, name);
    fscanf(file, "%lf\n", &ret);
    printf("read %f\n", ret);
    return(ret);
}

FILE* LOG;
static time_t TVAL;
void outlog(char* fmt, ...)
{
    if (LOG==NULL) { LOG=stdout; }

    TVAL = time(NULL);
    char* TIMESTR = ctime(&TVAL);
    TIMESTR[24]=' ';

    fprintf(LOG, "[ %20s] ", TIMESTR);

    va_list args;
    va_start(args, fmt);
    vfprintf(LOG, fmt, args);
    fprintf(LOG, "\n");
    va_end(args);
    fflush(LOG);
}

/*
 * resize a gsl vector
 *
 */


void resize(gsl_vector * vec, size_t newsize)
{
    assert(vec->stride == 1 && vec->owner && vec->block->data == vec->data);
    double * p = realloc(vec->block->data, newsize * sizeof(double));
    vec->block->data = vec->data = p;
    vec->size = newsize;
}

