#ifndef TOPICH
#define TOPICH

#include <gsl/gsl_vector.h>
#include <math.h>
#include "utils.h"
#include "typedefs.h"
#include "hyperparameter.h"

#define MH_ETA_STDEV 0.005
#define MH_GAM_STDEV 0.005

/* === topic methods === */

/*
 * update the count of a word in a topic
 *
 */

void topic_update_word(topic* t, int w, double update);


/*
 * update the document count in a topic
 *
 */

void topic_update_doc_cnt(topic* t, double update);


/*
 * write log probability in a file
 *
 */

void topic_write_log_prob(topic* t, FILE* f);


/* === tree methods === */

/*
 * allocate a new tree with a certain depth
 *
 */

tree* tree_new(int depth,
               int nwords,
               gsl_vector* eta,
               gsl_vector* gam,
               double scaling_shape,
               double scaling_scale);

/*
 * add children to a node down to the depth of the tree;
 * return the leaf of that path
 *
 */

topic* tree_fill(topic* t);


/*
 * add a child to a topic
 *
 */

topic* topic_add_child(topic* t);


/*
 * make a new topic
 *
 */

topic* topic_new(int nwords, int level, topic* parent, tree* tr);


/*
 * given a leaf with 0 instances, delete the leaf and all ancestors
 * which also have 0 instances.  (!!! use asserts here)
 *
 */

void tree_prune(topic* t);


/*
 * delete a node from the tree
 *
 */

void delete_node(topic* t);


/*
 * write a tree to file
 *
 */

void tree_write_log_prob(tree* tree, FILE* f);


/*
 * sample a document path from a tree
 *
 */

void populate_prob_dfs(topic* node, doc* d, double* logsum, double* pprob, int root_level);
void tree_sample_doc_path(tree* tr, doc* d, short do_remove, int root_level);

/*
 * sample a new path in the tree for a document
 *
 */

void tree_sample_path_for_doc(tree* t, doc* d);


/*
 * update the tree from an entire document
 *
 */

void tree_update_from_doc(doc* d, double update, int root_level);


/*
 * sample a leaf from the tree with populated probabilties
 *
 */

topic* tree_sample_path(topic* node, double logsum);
topic* tree_sample_dfs(double r, topic* node, double* sum, double logsum);
void tree_add_doc_to_path(topic* node, doc* d, int root_level);
void tree_remove_doc_from_path(tree* tr, doc* d, int root_level);

/*
 * write a tree to a file
 *
 */

void write_tree(tree* tf, FILE* file);
void write_tree_levels(tree* tr, FILE* file);
void write_tree_level_dfs(topic* t, FILE* f);
void write_tree_topics_dfs(topic* t, FILE* f);

/*
 * scores
 *
 */

double gamma_score(topic* t);
double gamma_score_PY(topic* t, double gam_add);
double eta_score(topic* t);
double log_gamma_ratio(doc* d, topic* t, int level);
double log_gamma_ratio_new(doc* d, int level, double eta, int nterms);
void tree_mh_update_eta(tree* tr);
void dfs_sample_scaling(topic* t);

/*
 * copying a tree
 *
 */

void copy_topic(const topic* src, topic* dest);
void copy_tree_dfs(const topic* src, topic* dest);
tree * copy_tree(const tree* tr);

void free_tree(tree * tr);
void free_tree_dfs(topic * t);

int ntopics_in_tree(tree * tr);
int ntopics_in_tree_dfs(topic * t);

#endif
