#include "topic.h"

/*
 * update the count of a word in a topic
 *
 */

void topic_update_word(topic* t, int w, double update)
{
    vinc(t->w_cnt, w, update);
    t->w_tot += update;

    // change log probability and log gammas

    double eta = vget(t->tr->eta, t->level);
    vset(t->log_prob_w, w,
         log(vget(t->w_cnt, w) + eta) -
         log(t->w_tot + t->w_cnt->size * eta));

    vset(t->lgam_w_plus_eta, w, lgam(vget(t->w_cnt, w) + eta));
}


/*
 * update the document count in a topic
 *
 */

void topic_update_doc_cnt(topic* t,
                          double update)
{
    t->doc_tot += update;
    t->log_doc_tot = log(t->doc_tot);
}


/*
 * topic score (i.e., eta score)
 *
 */

double eta_score(topic* t)
{
    int w, c;
    double score = 0;
    int nwords = t->w_cnt->size;
    double eta = vget(t->tr->eta, t->level);

    score = lgam(nwords * eta) - nwords * lgam(eta);

    for (w = 0; w < nwords; w++)
    {
	// score += lgam(vget(t->w_cnt, w) + eta);
        score += vget(t->lgam_w_plus_eta, w);
    }

    score -= lgam(t->w_tot + nwords * eta);
    // exponential(1) prior: log(1) - 1 * eta
    // score -= eta; (!!! note : this should not be added for each topic.)

    for (c = 0; c < t->nchild; c++)
    {
        if (t->child[c]->w_tot > 0)
            score += eta_score(t->child[c]);
    }
    return(score);
}


void tree_mh_update_eta(tree* tr)
{
    gsl_vector* vect = tr->eta;
    int depth = vect->size;
    int accept[depth];
    int iter;
    int l;
    for (l = 0; l < depth; l++) accept[l] = 0;

    for (iter = 0; iter < MH_REPS; iter++)
    {
        double current_score = eta_score(tr->root);
        for (l = 0; l < depth; l++)
        {
            double old = vget(vect, l);
            double new = rgauss(old, MH_ETA_STDEV);

            if (new > 0)
            {
                vset(vect, l, new);
                double new_score = eta_score(tr->root);
                double r = runif();
                if (r > exp(new_score - current_score))
                {
                    vset(vect, l, old);
                }
                else
                {
                    current_score = new_score;
                    accept[l]++;
                }
            }
        }
    }
    outlog("sampled eta");
}


/*
 * update the topics from a document beginning at a specified level
 *
 */

void tree_update_from_doc(doc* d, double update, int root_level)
{
    int depth = d->path[0]->tr->depth;
    int nword = d->word->size;
    int n;

    for (n = 0; n < nword; n++)
    {
        int level = ivget(d->levels, n);
        if (level > root_level)
            topic_update_word(d->path[level], ivget(d->word, n), update);
    }
    for (n = root_level + 1; n < depth; n++)
    {
        topic_update_doc_cnt(d->path[n], update);
    }
}


/*
 * sample a document path from a tree
 *
 */

// !!! these should be in doc.c

void tree_remove_doc_from_path(tree* tr, doc* d, int root_level)
{
    tree_update_from_doc(d, -1.0, root_level);
    tree_prune(d->path[tr->depth - 1]);
}


void tree_add_doc_to_path(topic* node, doc* d, int root_level)
{
    // set path

    int depth = node->tr->depth;
    int l = depth-1;
    do
    {
        d->path[l] = node;
        node = node->parent;
        l--;
    }
    while (l >= root_level);

    // update new path with this document

    tree_update_from_doc(d, +1.0, root_level);
}


/*
 * sample the path of a document starting from a particular level.
 *
 */

void tree_sample_doc_path(tree* tr,
                          doc* d,
                          short do_remove,
                          int root_level)
{
    // possibly remove document from path

    if (do_remove == 1)
    {
        tree_remove_doc_from_path(tr, d, root_level);
    }

    // compute probability

    double logsum = 0;
    double path_prob[tr->depth];
    populate_prob_dfs(d->path[root_level], d, &logsum, path_prob, root_level);

    // sample node and fill tree

    topic* node = tree_sample_path(d->path[root_level], logsum);
    node = tree_fill(node);

    // add document to this path

    tree_add_doc_to_path(node, d, root_level);
}


/*
 * populate the probability slot, based on a document
 *
 */

// !!! we can easily make these the same function and pass in an empty
// topic.

double log_gamma_ratio(doc* d, topic* t, int level)
{
    int nterms = t->log_prob_w->size;
    int nword = d->word->size;
    int n, count[nterms];
    double result;

    for (n = 0; n < nword; n++)
    {
	count[ivget(d->word, n)] = 0;
    }
    for (n = 0; n < nword; n++)
    {
	if (ivget(d->levels, n) == level)
	{
	    count[ivget(d->word, n)]++;
	}
    }

    double eta = vget(t->tr->eta, t->level);
    result = lgam(t->w_tot + nterms * eta); // !!! this should be precomputed
    result -= lgam(t->w_tot + vget(d->tot_levels, level) + nterms * eta);

    for (n = 0; n < nword; n++)
    {
        int wd = ivget(d->word, n);
        if (count[wd] > 0)
        {
            // result -= vget(t->lgam_w_plus_eta, wd);
            result -= lgam(vget(t->w_cnt, wd) + eta); // !!! this should be precomputed
            result += lgam(vget(t->w_cnt, wd) + count[wd] + eta);
            count[wd] = 0;
        }
    }
    return(result);
}


double log_gamma_ratio_new(doc* d, int level, double eta, int nterms)
{
    int n, count[nterms];
    double result;
    int nword = d->word->size;

    for (n = 0; n < nword; n++)
    {
	count[ivget(d->word, n)] = 0;
    }
    for (n = 0; n < nword; n++)
    {
	if (ivget(d->levels, n) == level)
	    count[ivget(d->word, n)]++;
    }
    result  = lgam(nterms*eta);
    result -= lgam(vget(d->tot_levels, level) + nterms * eta);

    for (n = 0; n < nword; n++)
    {
	int wd = ivget(d->word, n);
	if (count[wd] > 0)
	{
	    result -= lgam(eta);
	    result += lgam(count[wd] + eta);
	    count[wd] = 0;
	}
    }
    return(result);
}


void populate_prob_dfs(topic* node,
                       doc* d,
                       double* logsum,
                       double* pprob,
                       int root_level)
{
    int l, c;
    int level = node->level;
    int depth = node->tr->depth;

    // set path_prob for current node
    pprob[level]  = log_gamma_ratio(d, node, level);

    double denom = 0;
    if (level > root_level)
    {

        denom = log(node->parent->doc_tot + node->parent->scaling);
        // !!! PY process
        // denom = log(node->parent->doc_tot + node->parent->nchild * (level - 1) * 0.01 + vget(node->tr->gam, level - 1));
        // !!! possibly slowing us down
        // pprob[level] += node->log_doc_tot - denom;
        pprob[level] += log(node->doc_tot) - denom;
    }

    // set path probs for levels below this node
    // !!! do we need this if statement?
    if (level < depth - 1)
    {
        int nterms = node->log_prob_w->size;
        for (l = level+1; l < depth; l++)
        {
            double eta = vget(node->tr->eta, l);
            pprob[l] = log_gamma_ratio_new(d, l, eta, nterms);
        }
        // !!! PY process
        // double gam = vget(node->tr->gam, level) + node->nchild * level * 0.01;

        // !!! precompute these logs
        pprob[level+1] += log(node->scaling);
        pprob[level+1] -= log(node->doc_tot + node->scaling);
    }

    // set probability for this node
    node->prob = 0;
    for (l = root_level; l < depth; l++) node->prob += pprob[l];
    // printf("%d %10.7e\n", level, node->prob);

    // update the normalizing constant
    if (level==root_level)
        *logsum = node->prob;
    else
        *logsum = log_sum(*logsum, node->prob);

    // recurse
    for (c = 0; c < node->nchild; c++)
        populate_prob_dfs(node->child[c], d, logsum, pprob, root_level);
}



/*
 * prune tree
 *
 */

void tree_prune(topic* t)
{
    topic* parent = t->parent;
    if (t->doc_tot == 0)
    {
        delete_node(t);
        if (parent != NULL)
        {
            tree_prune(parent);
        }
    }
}


/*
 * delete a node from the tree
 *
 */

void delete_node(topic* t)
{
    int c;

    // delete all children
    for (c = 0; c < t->nchild; c++)
    {
	delete_node(t->child[c]);
    }

    // update parent
    int nc = t->parent->nchild;
    for (c = 0; c < nc; c++)
    {
	if (t->parent->child[c] == t)
	{
	    t->parent->child[c] = t->parent->child[nc - 1];
	    t->parent->nchild--;
	}
    }

    // free allocated memory for word counts and children
    gsl_vector_free(t->w_cnt);
    gsl_vector_free(t->log_prob_w);
    gsl_vector_free(t->lgam_w_plus_eta);
    free(t->child);
    free(t);
}


/*
 * fill tree
 *
 */

topic* tree_fill(topic* t)
{
    if (t->level < t->tr->depth-1)
    {
        topic* c = topic_add_child(t);
        return(tree_fill(c));
    }
    else
    {
        return(t);
    }
}


/*
 * add child
 *
 */

topic* topic_add_child(topic* t)
{
    // increase the number of children
    t->nchild++;
    // reallocate the child vector and create the new child
    t->child = (topic**) realloc(t->child, sizeof(topic*) * t->nchild);
    t->child[t->nchild - 1] = topic_new(t->w_cnt->size, t->level+1, t, t->tr);

    return(t->child[t->nchild - 1]);
}


/*
 * new topic
 *
 */

topic* topic_new(int nwords, int level, topic* parent, tree* tr)
{
    topic* t = malloc(sizeof(topic));

    t->w_tot = 0;
    t->w_cnt = gsl_vector_calloc(nwords);
    t->log_prob_w = gsl_vector_calloc(nwords);
    t->lgam_w_plus_eta = gsl_vector_calloc(nwords);
    t->log_doc_tot = 0; // !!! make this a NAN?
    t->doc_tot = 0;
    t->level = level;
    t->nchild = 0;
    t->child = NULL;
    t->parent = parent;
    t->tr = tr;
    t->id = tr->next_id++;
    // sample the scaling parameter from the prior
    // !!! here we can set it to the level gamma to reproduce the old code
    // t->scaling = rgamma(tr->scaling_shape, tr->scaling_scale);
    t->scaling = t->tr->scaling_shape * t->tr->scaling_scale;

    // set log probabilities

    double eta = vget(t->tr->eta, t->level);
    double log_p_w = log(eta) - log(eta * nwords);
    gsl_vector_set_all(t->log_prob_w, log_p_w);

    return(t);
}


/*
 * sample_node draws a random number and then selects a node in the
 * tree based on prob and that random number.  it takes the log
 * normalizer as an argument and normalizes the probabilities as it
 * goes through them.
 *
 */

topic* tree_sample_path(topic* root, double logsum)
{

    double running_sum = 0;
    double r = runif();
    // outlog("rand=%7.5f", r);
    return(tree_sample_dfs(r, root, &running_sum, logsum));
}


/*
 * r       : random number
 * node    : current node
 * lognorm : log normalizer
 * sum     : pointer to running sum -- updated at each call
 *
 */

topic* tree_sample_dfs(double r, topic* node, double* sum, double logsum)
{
    *sum = *sum + exp(node->prob - logsum);
    if (*sum >= r)
    {
        // outlog("selected node at level%d with prob %7.5e",
        // node->level, exp(node->prob-logsum));
	return(node);
    }
    else
    {
        int i;
	for (i = 0; i < node->nchild; i++)
	{
	    topic* val = tree_sample_dfs(r, node->child[i], sum, logsum);
	    if (val != NULL)
	    {
		return(val);
	    }
	}
    }
    return(NULL);
}


/*
 * tree score (i.e., gamma score)
 * includes a flag to decide whether to sample the scaling parameter
 *
 */

// !!! is this right even if we have empty topics that aren't used?

double gamma_score(topic* t)
{
    int c;
    double score = 0;

    if (t->nchild > 0)
    {
        // !!! this is only appropriate when we have a prior on gamma
        // score += log_dgamma(t->scaling,
        // t->tr->scaling_shape,
        // t->tr->scaling_scale);
        // score += log(t->scaling) * t->nchild; // !!! what is this?
        score -= lgam(t->scaling + t->doc_tot);
        for (c = 0; c < t->nchild; c++)
        {
            score += lgam(t->scaling + t->child[c]->doc_tot);
            score += gamma_score(t->child[c]);
        }
    }
    return(score);
}


/*
 * recursively sample the scaling parameters
 *
 */

void dfs_sample_scaling(topic* t)
{
    if (t->nchild > 0)
    {
        t->scaling = gibbs_sample_DP_scaling(t->scaling,
                                             t->tr->scaling_shape,
                                             t->tr->scaling_scale,
                                             t->nchild,
                                             t->doc_tot);
    }
    int c;
    for (c = 0; c < t->nchild; c++)
    {
        dfs_sample_scaling(t->child[c]);
    }
}



/*
 * allocate a new tree
 *
 */

tree* tree_new(int depth,
               int nwords,
               gsl_vector* eta,
               gsl_vector* gam,
               double scaling_shape,
               double scaling_scale)
{
    tree* tr    = malloc(sizeof(tree));

    tr->depth   = depth;
    tr->eta     = eta;
    tr->gam     = gam;
    tr->next_id = 0;
    tr->scaling_shape = scaling_shape;
    tr->scaling_scale = scaling_scale;

    tr->root = topic_new(nwords, 0, NULL, tr);

    return(tr);
}


/*
 * write a topic tree
 *
 */

void write_tree_topics_dfs(topic* root_topic, FILE* file)
{
    int i;

    fprintf(file, "%-6d", root_topic->id);

    if (root_topic->parent != NULL)
        fprintf(file, " %-6d", root_topic->parent->id);
    else
        fprintf(file, " %-6d", -1);

    fprintf(file, " %06.0f", root_topic->doc_tot);
    fprintf(file, " %06.0f", root_topic->w_tot);
    fprintf(file, " %06.3e", root_topic->scaling);

    for (i = 0; i < root_topic->w_cnt->size; i++)
    {
        fprintf(file, " %6.0f", vget(root_topic->w_cnt, i));
    }
    fprintf(file, "\n");

    for (i = 0; i < root_topic->nchild; i++)
    {
        write_tree_topics_dfs(root_topic->child[i], file);
    }
}

/*
 * compute the number of topics in a tree
 *
 */

int ntopics_in_tree(tree * tr)
{
    return(ntopics_in_tree_dfs(tr->root));
}

int ntopics_in_tree_dfs(topic * t)
{
    int topics_below = 0;
    int c;
    for (c = 0; c < t->nchild; c++)
    {
        topics_below += ntopics_in_tree_dfs(t->child[c]);
    }
    return(t->nchild + topics_below);
}


/*
 * free a tree
 *
 */

void free_tree(tree * tr)
{
    free_tree_dfs(tr->root);
    free(tr);
}

void free_tree_dfs(topic * t)
{
    gsl_vector_free(t->w_cnt);
    gsl_vector_free(t->log_prob_w);
    gsl_vector_free(t->lgam_w_plus_eta);
    int c;
    for (c = 0; c < t->nchild; c++)
        free_tree_dfs(t->child[c]);
    free(t->child);
    free(t);
}

/*
 * copy a tree
 *
 */

tree * copy_tree(const tree* tr)
{
    tree* tree_copy = tree_new(tr->depth,
                               tr->root->w_cnt->size,
                               tr->eta,
                               tr->gam,
                               tr->scaling_shape,
                               tr->scaling_scale);

    copy_tree_dfs(tr->root, tree_copy->root);

    return(tree_copy);
}

void copy_tree_dfs(const topic* src, topic* dest)
{
    copy_topic(src, dest);

    int c;
    for (c = 0; c < src->nchild; c++)
    {
        topic* child = topic_add_child(dest);
        child->parent = dest;
        copy_tree_dfs(src->child[c], child);
    }
}

void copy_topic(const topic* src, topic* dest)
{
    dest->w_tot = src->w_tot;
    gsl_vector_memcpy(dest->w_cnt, src->w_cnt);
    gsl_vector_memcpy(dest->log_prob_w, src->log_prob_w);
    gsl_vector_memcpy(dest->lgam_w_plus_eta, src->lgam_w_plus_eta);
    dest->doc_tot = src->doc_tot;
    dest->log_doc_tot = src->log_doc_tot;
    dest->id = src->id;
    dest->level = src->level;
    dest->nchild = 0; // children get added separately
    dest->scaling = src->scaling;
    dest->prob = src->prob;
}


/*
 * write levels of the tree
 *
 */

void write_tree_levels(tree* tr, FILE* file)
{
    write_tree_level_dfs(tr->root, file);
    fprintf(file, "\n");
    fflush(file);
}


void write_tree_level_dfs(topic* root_topic, FILE* file)
{
    int i;
    if (root_topic->parent == NULL)
    {
        fprintf(file, "%d", root_topic->level);
    }
    else
    {
        fprintf(file, " %d", root_topic->level);
    }
    for (i = 0; i < root_topic->nchild; i++)
    {
        write_tree_level_dfs(root_topic->child[i], file);
    }
}


void write_tree(tree* tf, FILE* file)
{
    fprintf(file, "%-6s %-6s %-6s %-6s  %-9s %-6s\n",
            "ID", "PARENT", "NDOCS", "NWORDS", "SCALE", "WD_CNT");
    write_tree_topics_dfs(tf->root, file);
}
