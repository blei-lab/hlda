#ifndef DOCH
#define DOCH

#include <gsl/gsl_vector.h>
#include <assert.h>
#include "utils.h"
#include "topic.h"
#include "typedefs.h"

#define OFFSET 0
#define MH_GEM_STDEV 0.05
#define MH_GEM_MEAN_STDEV 0.05
#define MH_GEM_STDEV 0.05

/*
 * resample the levels of a document
 *
 */

void doc_sample_levels(doc* d, short do_permute, short do_remove);

/*
 * update a level count
 *
 */

void doc_update_level(doc* d, int l, double update);


/*
 * read corpus from data
 *
 */

void read_corpus(char* filename, corpus* c, int depth);

/*
 * allocate a new corpus
 *
 */

corpus* corpus_new(double mean, double scale);

/*
 * score the corpus
 *
 */

double gem_score(corpus* corp);

/*
 * GEM MH updates
 *
 */

void corpus_mh_update_gem(corpus* corp);
void corpus_mh_update_gem_mean(corpus* corp);
void corpus_mh_update_gem_scale(corpus* corp);

void compute_log_p_level(doc* d, double gem_mean, double gem_scale);

/*
 * write the document clustering to a file
 *
 */

void write_corpus_assignment(corpus* corp, FILE* file);
void write_corpus_levels(corpus* corp, FILE* file);

// free a corpus

void free_corpus(corpus* corp);
void free_doc(doc* d);

#endif
