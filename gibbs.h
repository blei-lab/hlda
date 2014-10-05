#ifndef GIBBSH
#define GIBBSH

#include "utils.h"
#include "typedefs.h"
#include "doc.h"
#include "topic.h"

#include <stdio.h>

#define WRITE_MODE_CORPUS 1
#define DEFAULT_OUTPUT_LAG 1000
#define DEFAULT_HYPER_LAG 1
#define DEFAULT_SHUFFLE_LAG 100
#define DEFAULT_LEVEL_LAG -1
#define DEFAULT_SAMPLE_GAM 0
#define NINITREP 100

void write_gibbs_state(gibbs_state * state, char* filename);

void write_gibbs_output(gibbs_state * state);

void compute_gibbs_score(gibbs_state * state);

void iterate_gibbs_state(gibbs_state * state);

gibbs_state * new_gibbs_state(char* corpus, char* settings);

gibbs_state * new_heldout_gibbs_state(corpus* corp, gibbs_state* orig);

double mean_heldout_score(corpus* corp,
                          gibbs_state* orig,
                          int burn,
                          int lag,
                          int niter);

void free_gibbs_state(gibbs_state* state);

void init_gibbs_state(gibbs_state * state);

gibbs_state * init_gibbs_state_w_rep(char* corpus,
                                     char* settings,
                                     char* out_dir);

void set_up_directories(gibbs_state * state, char * out_dir);

#endif
