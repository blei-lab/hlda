#ifndef TYPEDEFS
#define TYPEDEFS

#define MH_REPS 100
#define TRUE 1
#define FALSE 0

#include <gsl/gsl_vector.h>
#include "utils.h"

typedef struct topic
{
    double w_tot;                // total # of words assigned to the topic
    gsl_vector* w_cnt;           // vector of word counts
    gsl_vector* log_prob_w;      // vector of log probabilities
    gsl_vector* lgam_w_plus_eta; // to prevent many repeated computations

    double doc_tot;         // total number of doc-level instances
    double log_doc_tot;     // precomputed log of doc_tot
    int id;                 // a unique ID for this topic

    int level;              // level in the tree
    int nchild;             // number of children
    double scaling;         // scaling factor for my DP
    struct topic** child;   // array of pointers to child topics
    struct topic* parent;   // pointer to the parent topic
    struct tree* tr;        // pointer to my tree

    double prob;            // probability (used to sample a path)
} topic;


typedef struct tree
{
    int depth;              // depth of the tree
    gsl_vector* eta;        // topic dirichlet parameter
    gsl_vector* gam;        // scaling parameter (!!! not used; see G prior)
    double scaling_shape;   // shape parameter for the G prior on scaling
    double scaling_scale;   // scale parameter for the G prior on scaling
    topic* root;            // root topic of the tree
    int next_id;            // the next id for a new topic
} tree;


typedef struct doc
{
    int_vector* word;            // each word
    int_vector* levels;          // level assigned to each word

    int id;
    topic** path;                // path of topics
    gsl_vector* tot_levels;      // level counts (convenience, for sampling)
    gsl_vector* log_p_level;     // log p(level) [ unnormalized ]
    double* gem_mean;
    double* gem_scale;
    double score;
} doc;


typedef struct corpus
{
    double gem_mean;
    double gem_scale;
    int ndoc;
    int nterms;
    doc** doc;
} corpus;

typedef struct gibbs_state
{
    // data and hidden variables
    corpus* corp;
    tree* tr;

    // current scores and iteration
    double score;
    double gem_score;
    double eta_score;
    double gamma_score;
    double max_score;
    int iter;

    // log files
    char* run_dir;
    FILE* score_log;
    FILE* tree_structure_log;

    // sampling parameters
    int shuffle_lag;
    int hyper_lag;
    int level_lag;
    int output_lag;
    int sample_eta;
    int sample_gem;
    int sample_gam;
} gibbs_state;

#endif
