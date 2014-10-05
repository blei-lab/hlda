#include "doc.h"

/*
 * resample the levels of a document
 *
 */

void doc_sample_levels(doc* d,
                       short do_permute,
                       short do_remove)
{
    int i;
    int depth = d->path[0]->tr->depth;
    gsl_vector* log_prob = gsl_vector_alloc(depth);

    if (do_permute == 1)
    {
        gsl_permutation* p = rpermutation(d->levels->size);
        iv_permute_from_perm(d->levels, p);
        iv_permute_from_perm(d->word, p);
        gsl_permutation_free(p);
    }

    // resample levels

    for (i = 0; i < d->word->size; i++)
    {
        int w = ivget(d->word, i);
        if (do_remove == 1)
        {
            int l = ivget(d->levels, i);
            doc_update_level(d, l, -1.0);
            topic_update_word(d->path[l], w, -1.0);
        }

        // compute probabilties

        int k;
        compute_log_p_level(d, *(d->gem_mean), *(d->gem_scale));
        for (k = 0; k < depth; k++)
        {
            vset(log_prob, k,
                 vget(d->log_p_level, k) +
                 vget(d->path[k]->log_prob_w, w));
        }

        // sample new level and update

        int new_l = sample_from_log(log_prob);
        topic_update_word(d->path[new_l], w, 1.0);
        ivset(d->levels, i, new_l);
        // !!! this should take the word position, and remove or add it
        doc_update_level(d, new_l, 1.0);
    }

    gsl_vector_free(log_prob);
}


/*
  compute the log probability of each level, conditioned on the
  current level counts
*/

void compute_log_p_level(doc* d, double gem_mean, double gem_scale)
{
    // first, compute E[stick size]

    double levels_sum = sum(d->tot_levels);
    double sum_log_prob = 0;
    double last_section = 0;
    int i;

    for (i = 0; i < d->tot_levels->size-1; i++)
    {
        levels_sum -= vget(d->tot_levels, i);

        double expected_stick_len =
            ((1 - gem_mean) * gem_scale + vget(d->tot_levels, i)) /
            (gem_scale + vget(d->tot_levels, i) + levels_sum);

        vset(d->log_p_level,
             i,
             log(expected_stick_len) + sum_log_prob);

        if (i == 0)
            last_section = vget(d->log_p_level, i);
        else
            last_section = log_sum(vget(d->log_p_level, i), last_section);

        sum_log_prob += log(1 - expected_stick_len);
    }
    last_section = log(1.0 - exp(last_section));
    vset(d->log_p_level, d->tot_levels->size-1, last_section);
}

/*
 * update the level counts
 *
 */

void doc_update_level(doc* d, int l, double update)
{
    vinc(d->tot_levels, l, update);
}


/*
 * read corpus from data
 *
 */

void read_corpus(char* data_filename, corpus* c, int depth)
{
    outlog("READING CORPUS FROM %s", data_filename);

    FILE *fileptr;
    int nunique, count, word, n, i, total = 0;
    doc *d;
    c->nterms = 0;
    c->ndoc = 0;

    fileptr = fopen(data_filename, "r");

    while (fscanf(fileptr, "%10d", &nunique) != EOF)
    {
        c->ndoc = c->ndoc + 1;

        if ((c->ndoc % 100) == 0) outlog("read doc %d", c->ndoc);

        c->doc = (doc**) realloc(c->doc, sizeof(doc*) * c->ndoc);
        c->doc[c->ndoc-1] = malloc(sizeof(doc));
        d = c->doc[c->ndoc-1];
        d->id = c->ndoc-1;
        d->word = new_int_vector(0);

        // read document

        for (n = 0; n < nunique; n++)
        {
            fscanf(fileptr, "%10d:%10d", &word, &count);
            total += count;
            word = word - OFFSET;
            assert(word >= 0);

            if (word >= c->nterms)
            {
                c->nterms = word + 1;
            }
            for (i = 0; i < count; i++)
            {
                ivappend(d->word, word);
            }
        }

        // set up gibbs state variables

        d->levels      = new_int_vector(d->word->size);
        d->path        = malloc(sizeof(topic*) * depth);
        d->tot_levels  = gsl_vector_calloc(depth);
        d->log_p_level = gsl_vector_calloc(depth);
        d->gem_mean    = &(c->gem_mean);
        d->gem_scale   = &(c->gem_scale);

        for (n = 0; n < d->levels->size; n++)
            ivset(d->levels, n, -1);
    }

    fclose(fileptr);
    outlog("number of docs    : %d", c->ndoc);
    outlog("number of words   : %d", c->nterms);
    outlog("total word count  : %d", total);
}


/*
 * allocate a new corpus
 *
 */

corpus* corpus_new(double gem_mean, double gem_scale)
{
    corpus* c = malloc(sizeof(corpus));

    c->gem_mean = gem_mean;
    c->gem_scale = gem_scale;
    c->ndoc = 0;
    c->doc = malloc(sizeof(doc*) * c->ndoc);

    return(c);
}

void free_corpus(corpus* corp)
{
    int d;
    for (d = 0; d < corp->ndoc; d++)
    {
        free_doc(corp->doc[d]);
    }
    free(corp->doc);
    free(corp);
}

void free_doc(doc* d)
{
    delete_int_vector(d->word);
    delete_int_vector(d->levels);
    gsl_vector_free(d->tot_levels);
    gsl_vector_free(d->log_p_level);
    free(d);
}

/*
 * write corpus assignment
 * each line contains a space delimited list of topic IDs
 *
 */

void write_corpus_assignment(corpus* corp, FILE* file)
{
    int d, l;
    int depth = corp->doc[0]->path[0]->tr->depth;
    for (d = 0; d < corp->ndoc; d++)
    {
        fprintf(file, "%d", corp->doc[d]->id);
        fprintf(file, " %1.9e", (corp->doc[d]->score /
                                 (double) corp->doc[d]->word->size));
        for (l = 0; l < depth; l++)
        {
            fprintf(file, " %d", corp->doc[d]->path[l]->id);
        }
        fprintf(file, "\n");
    }
}


void write_corpus_levels(corpus* corp, FILE* file)
{
    outlog("writing all corpus level variables");
    int d, n;
    for (d = 0; d < corp->ndoc; d++)
    {
        for (n = 0; n < corp->doc[d]->word->size; n++)
        {
            if (n > 0) fprintf(file, " ");
            fprintf(file, "%d:%d",
                    ivget(corp->doc[d]->word, n),
                    ivget(corp->doc[d]->levels, n));
        }
        fprintf(file, "\n");
    }
}


/*
 * corpus score (i.e., GEM score)
 *
 */

double gem_score(corpus* corp)
{
    double score = 0;
    int depth = corp->doc[0]->path[0]->tr->depth;
    double prior_a = (1 - corp->gem_mean) * corp->gem_scale;
    double prior_b = corp->gem_mean * corp->gem_scale;
    int i, l, k;
    for (i = 0; i < corp->ndoc; i++)
    {
        doc* curr_doc = corp->doc[i];
        curr_doc->score = 0;
        double count_gt_k[depth];
        for (l = 0; l < depth; l++)
        {
            count_gt_k[l] = 0;
            double count = vget(curr_doc->tot_levels, l);
            for (k = 0; k < l; k++)
                count_gt_k[k] += count;
        }
        double sum_log_prob = 0;
        double levels_sum = sum(curr_doc->tot_levels);
        double last_log_prob = 0;
        for (l = 0; l < depth-1; l++)
        {
            double a = vget(curr_doc->tot_levels, l) + prior_a;
            double b = count_gt_k[l] + prior_b;
            curr_doc->score +=
                lgamma(a) + lgamma(b) - lgamma(a + b) -
                lgamma(prior_b) - lgamma(prior_a) +
                lgamma(prior_a + prior_b);

            // compute the probability of this level for computing the
            // probability of the bottom level later.

            levels_sum -= vget(curr_doc->tot_levels, l);
            double expected_stick_len =
                (prior_a + vget(curr_doc->tot_levels, l)) /
                (corp->gem_scale + vget(curr_doc->tot_levels, l) + levels_sum);

            double log_p = log(expected_stick_len) + sum_log_prob;
            if (l==0)
                last_log_prob = log_p;
            else
                last_log_prob += log_sum(log_p, last_log_prob);
            sum_log_prob += log(1 - expected_stick_len);
        }
        last_log_prob = log(1 - exp(last_log_prob));

        // now handle the bottom levels, which are conditionally
        // independent given everything else.  (more z's allocated to
        // the last level doesn't make other's any more likely because the
        // probability of reaching the last level has only to do with the
        // previous stick lenths.)

        curr_doc->score += vget(curr_doc->tot_levels, depth-1) * last_log_prob;

        score += curr_doc->score;
    }
    // exponential 1 prior: log(1) - 1 * s
    score += -corp->gem_scale;
    return(score);
}


void corpus_mh_update_gem(corpus* corp)
{
    double current_score = gem_score(corp);

    int accept = 0;
    int iter;
    for (iter = 0; iter < MH_REPS; iter++)
    {
        double old_mean = corp->gem_mean;
        double old_scale = corp->gem_scale;
        double old_alpha = corp->gem_mean * corp->gem_scale;
        double new_alpha = rgauss(old_alpha, MH_GEM_STDEV);
        double new_mean = new_alpha / (1.0 + new_alpha);
        double new_scale = 1.0 + new_alpha;

        if (new_alpha < 0) continue;

        corp->gem_mean = new_mean;
        corp->gem_scale = new_scale;
        double new_score = gem_score(corp);
        double r = runif();
        if (r > exp(new_score - current_score))
        {
          corp->gem_mean = old_mean;
          corp->gem_scale = old_scale;
        }
        else
        {
          current_score = new_score;
          accept++;
        }
    }
    outlog("sampled gem: accepted %d; mean = %5.3f scale = %5.3f",
           accept, corp->gem_mean, corp->gem_scale);
}


void corpus_mh_update_gem_mean(corpus* corp)
{
    outlog("updating gem");
    double current_score = gem_score(corp);

    int accept = 0;
    int iter;
    for (iter = 0; iter < MH_REPS; iter++)
    {
        double old_mean = corp->gem_mean;
        double new_mean = rgauss(old_mean, MH_GEM_MEAN_STDEV);

        if ((new_mean > 0) && (new_mean < 1))
        {
            corp->gem_mean = new_mean;
            double new_score = gem_score(corp);
            double r = runif();
            if (r > exp(new_score - current_score))
            {
                corp->gem_mean = old_mean;
            }
            else
            {
                current_score = new_score;
                accept++;
            }
        }
    }
    outlog("sampled gem mean [accepted %d; mean = %5.3f]",
           accept, corp->gem_mean);
}

void corpus_mh_update_gem_scale(corpus* corp)
{
    // outlog("updating gem");
    double current_score = gem_score(corp);

    int accept = 0;
    int iter;
    for (iter = 0; iter < MH_REPS; iter++)
    {
        double old_scale = corp->gem_scale;
        double new_scale = rgauss(old_scale, MH_GEM_STDEV);

        if (new_scale > 0)
        {
            corp->gem_scale = new_scale;
            double new_score = gem_score(corp);
            double r = runif();
            if (r > exp(new_score - current_score))
            {
                corp->gem_scale = old_scale;
            }
            else
            {
                current_score = new_score;
                accept++;
            }
        }
    }
    outlog("sampled gem scale [ accepted %d; scale = %5.3f]",
           accept, corp->gem_scale);
}
