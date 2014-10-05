#include "gibbs.h"

extern gsl_rng* RANDNUMGEN;

void write_gibbs_state(gibbs_state * state, char* filename)
{
    tree * tr = state->tr;
    corpus * corp = state->corp;
    double score = state->score;

    char topic_filename[100];
    sprintf(topic_filename, "%s.topics", filename);
    FILE* file = fopen(filename, "w");
    write_double(score, "SCORE", file);
    write_int(state->iter, "ITER", file);
    write_vect(tr->eta, "ETA", file);
    write_vect(tr->gam, "GAMMA", file);
    write_double(corp->gem_mean, "GEM_MEAN", file);
    write_double(corp->gem_scale, "GEM_SCALE", file);
    write_double(tr->scaling_shape, "SCALING_SHAPE", file);
    write_double(tr->scaling_scale, "SCALING_SCALE", file);

    write_tree(tr, file);
    char assign_filename[100];
    sprintf(assign_filename, "%s.assign", filename);
    FILE* assign_file = fopen(assign_filename, "w");
    write_corpus_assignment(corp, assign_file);
    fclose(assign_file);
    fclose(file);
}


void write_gibbs_output(gibbs_state * state)
{
    FILE* score_f = state->score_log;
    corpus * corp = state->corp;
    tree * tr = state->tr;
    int depth = tr->depth;

    if (score_f != NULL)
    {
        fprintf(state->score_log,
                "%06d %14.3f %14.3f %14.3f %14.3f %7.4e %7.4e",
                state->iter, state->gem_score, state->eta_score,
                state->gamma_score, state->score,
                corp->gem_mean, corp->gem_scale);
        int l;
        for (l = 0; l < depth - 1; l++)
        {
            fprintf(state->score_log, " %7.4e", vget(tr->gam,l));
        }
        for (l = 0; l < depth; l++)
        {
            fprintf(state->score_log, " %7.4e", vget(tr->eta,l));
        }

        fprintf(state->score_log, "\n");
        fflush(state->score_log);
    }
    if (state->tree_structure_log != NULL)
    {
        write_tree_levels(tr, state->tree_structure_log);
    }
    if (state->run_dir != NULL)
    {
        char filename[100];
        if ((state->output_lag > 0) &&
            (state->iter % state->output_lag) == 0)
        {
            sprintf(filename, "%s/iter=%06d", state->run_dir, state->iter);
            write_gibbs_state(state, filename);
        }
        if (state->score == state->max_score)
        {
            outlog("mode at iteration %04d", state->iter);
            sprintf(filename, "%s/mode", state->run_dir);
            write_gibbs_state(state, filename);
            sprintf(filename, "%s/mode.levels", state->run_dir);
            FILE* levels_file = fopen(filename, "w");
            write_corpus_levels(state->corp, levels_file);
            fclose(levels_file);
        }
    }
}

void compute_gibbs_score(gibbs_state * state)
{
    tree * tr = state->tr;
    corpus * corp = state->corp;

    state->gem_score = gem_score(corp);
    state->eta_score = eta_score(tr->root);
    state->gamma_score = gamma_score(tr->root);
    state->score = state->gem_score + state->eta_score + state->gamma_score;
    if ((state->score > state->max_score) || (state->iter == 0))
    {
        state->max_score = state->score;
    }
}

void iterate_gibbs_state(gibbs_state * state)
{
    tree* tr = state->tr;
    corpus* corp = state->corp;
    state->iter = state->iter + 1;
    int iter = state->iter;
    outlog("iteration %04d (%04d topics)",
           iter, ntopics_in_tree(state->tr));

    // set up the sampling level (or fix at the depth - 1)
    int sampling_level = 0;
    if (state->level_lag == -1)
    {
        sampling_level = 0;
    }
    else if ((iter % state->level_lag) == 0)
    {
        int level_inc = iter / state->level_lag;
        sampling_level = level_inc % (tr->depth - 1);
        outlog("sampling at level %d", sampling_level);
    }
    // set up shuffling
    int do_shuffle = 0;
    if (state->shuffle_lag > 0)
    {
       do_shuffle = 1 - (iter % state->shuffle_lag);
    }
    if (do_shuffle == TRUE)
    {
        gsl_ran_shuffle(RANDNUMGEN, corp->doc, corp->ndoc, sizeof(doc*));
    }
    // sample paths and level allocations
    int d;
    for (d = 0; d < corp->ndoc; d++)
    {
        tree_sample_doc_path(tr, corp->doc[d], 1, sampling_level);
    }
    for (d = 0; d < corp->ndoc; d++)
    {
        doc_sample_levels(corp->doc[d], do_shuffle, 1);
    }
    // sample hyper-parameters
    if ((state->hyper_lag > 0) && (iter % state->hyper_lag) == 0)
    {
        if (state->sample_eta == 1)
        {
            tree_mh_update_eta(tr);
        }
        if (state->sample_gem == 1)
        {
            corpus_mh_update_gem_scale(corp);
            corpus_mh_update_gem_mean(corp);
        }
        if (state->sample_gam)
        {
            dfs_sample_scaling(tr->root);
            // tree_mh_update_gam(tr);
        }
    }
    compute_gibbs_score(state);
    write_gibbs_output(state);
}

void init_gibbs_state(gibbs_state* state)
{
    tree* tr = state->tr;
    corpus* corp = state->corp;
    gsl_ran_shuffle(RANDNUMGEN, corp->doc, corp->ndoc, sizeof(doc*));
    int depth = tr->depth;
    int i, j;
    for (i = 0; i < corp->ndoc; i++)
    {
        doc* d = corp->doc[i];
        gsl_vector_set_zero(d->tot_levels);
        gsl_vector_set_zero(d->log_p_level);
        iv_permute(d->word);
        d->path[depth - 1] = tree_fill(tr->root);
        topic_update_doc_cnt(d->path[depth - 1], 1.0);
        for (j = depth - 2; j >= 0; j--)
        {
            d->path[j] = d->path[j+1]->parent;
            topic_update_doc_cnt(d->path[j], 1.0);
        }
        doc_sample_levels(d, 0, 0);
        if (i > 0) tree_sample_doc_path(tr, d, 1, 0);
        doc_sample_levels(d, 0, 1);
    }
    compute_gibbs_score(state);
}

gibbs_state* init_gibbs_state_w_rep(char* corpus_fname,
                                    char* settings,
                                    char* out_dir)
{
    outlog("initializing state");
    gibbs_state* best_state;
    double best_score = 0;
    int rep;
    for (rep = 0; rep < NINITREP; rep++)
    {
        gibbs_state* state = new_gibbs_state(corpus_fname,
                                             settings);
        init_gibbs_state(state);

        if ((rep == 0) || (state->score > best_score))
        {
            outlog("best initial state at rep = %03d; score = %10.7e",
                   rep, state->score);
            best_state = state;
            best_score = state->score;
        }
        else
        {
            free_gibbs_state(state);
        }
    }
    if (out_dir != NULL)
    {
        set_up_directories(best_state, out_dir);
        char filename[100];
        sprintf(filename, "%s/initial", best_state->run_dir);
        write_gibbs_state(best_state, filename);
        sprintf(filename, "%s/mode", best_state->run_dir);
        write_gibbs_state(best_state, filename);
    }
    outlog("done initializing state");
    return(best_state);
}


gibbs_state * new_gibbs_state(char* corpus, char* settings)
{
    gibbs_state * state = malloc(sizeof(gibbs_state));
    init_random_number_generator();

    // read hyperparameters
    FILE* init = fopen(settings, "r");
    int depth = read_int("DEPTH", init);
    gsl_vector* eta = read_vect("ETA", depth, init);
    gsl_vector* gam = read_vect("GAM", depth - 1, init);
    double gem_mean = read_double("GEM_MEAN", init);
    double gem_scale = read_double("GEM_SCALE", init);
    double scaling_shape = read_double("SCALING_SHAPE", init);
    double scaling_scale = read_double("SCALING_SCALE", init);
    int sample_eta = read_int("SAMPLE_ETA", init);
    int sample_gem = read_int("SAMPLE_GEM", init);

    // set up the gibbs state
    state->iter = 0;
    state->corp = corpus_new(gem_mean, gem_scale);
    read_corpus(corpus, state->corp, depth);
    state->tr = tree_new(depth, state->corp->nterms,
                         eta,
                         gam,
                         scaling_shape,
                         scaling_scale);

    state->shuffle_lag = DEFAULT_SHUFFLE_LAG;
    state->hyper_lag   = DEFAULT_HYPER_LAG;
    state->level_lag   = DEFAULT_LEVEL_LAG;
    state->output_lag  = DEFAULT_OUTPUT_LAG;
    state->sample_eta  = sample_eta;
    state->sample_gem  = sample_gem;
    state->sample_gam  = DEFAULT_SAMPLE_GAM;
    state->run_dir = NULL;
    state->score_log = NULL;
    state->tree_structure_log = NULL;
    return(state);
}


void set_up_directories(gibbs_state * state, char * out_dir)
{
    state->run_dir = malloc(sizeof(char) * 100);
    // set up the run directory
    int id = 0;
    sprintf(state->run_dir, "%s/run%03d", out_dir, id);
    while (directory_exist(state->run_dir))
    {
        id++;
        sprintf(state->run_dir, "%s/run%03d", out_dir, id);
    }
    mkdir(state->run_dir, S_IRUSR|S_IWUSR|S_IXUSR);
    // set up  output files
    char filename[100];
    sprintf(filename, "%s/tree.log", state->run_dir);
    state->tree_structure_log = fopen(filename, "w");
    sprintf(filename, "%s/score.log", state->run_dir);
    state->score_log = fopen(filename, "w");
    fprintf(state->score_log,
            "%6s %14s %14s %14s %14s %10s %10s",
            "iter", "gem.score", "eta.score", "gamma.score",
            "total.score", "gem.mean", "gem.scale");
    int l;
    for (l = 0; l < state->tr->depth - 1; l++)
        fprintf(state->score_log, " %8s.%d", "gamma", l);
    for (l = 0; l < state->tr->depth; l++)
        fprintf(state->score_log, " %8s.%d", "eta", l);
    fprintf(state->score_log, "\n");
    fflush(state->score_log);
}


/*
  gibbs_state * parcopy_gibbs_state(gibbs_state* orig)
  {
  gibbs_state * state = malloc(sizeof(gibbs_state));
  state->corp = copy_corp(orig->corp);
  state->tr = copy_tree(orig->tr);
  state->run_dir = orig->run_dir;
  state->score_log = orig->score_log;
  state->tree_structure_log = orig->tree_structure_log;
  state->shuffle_lag = orig->shuffle_lag;
  state->hyper_lag = orig->hyper_lag;
  state->level_lag = orig->level_lag;
  state->output_lag = orig->output_lag;
  state->iter = orig->iter;
  }
*/


gibbs_state * new_heldout_gibbs_state(corpus* corp, gibbs_state* orig)
{
    gibbs_state * state = malloc(sizeof(gibbs_state));
    state->corp = corp;
    state->tr = copy_tree(orig->tr);

    state->run_dir = NULL;
    state->score_log = NULL;
    state->tree_structure_log = NULL;

    state->shuffle_lag = orig->shuffle_lag;
    state->hyper_lag = -1;
    state->level_lag = orig->level_lag;
    state->output_lag = -1;
    state->iter = 0;

    return(state);
}


double mean_heldout_score(corpus* corp,
                          gibbs_state* orig,
                          int burn,
                          int lag,
                          int niter)
{
    double score = 0;
    int nsamples = 0;

    gibbs_state* state = new_heldout_gibbs_state(corp, orig);

    init_gibbs_state(state);
    int iter = 0;
    for (iter = 0; iter < niter; iter++)
    {
        if ((iter % 100) == 0) outlog("held-out iter %04d", iter);
        iterate_gibbs_state(state);
        if ((iter >= burn) && ((iter % lag) == 0))
        {
            double this_score = state->score - orig->score;
            this_score -= state->gamma_score;
            this_score += orig->gamma_score;
            score += this_score;
            nsamples += 1;
        }
    }
    score = score / nsamples;
    outlog("mean held-out score = %7.3f (%d samples)", score, nsamples);
    free_tree(state->tr);
    free(state);
    return(score);
}


void free_gibbs_state(gibbs_state* state)
{
    free_corpus(state->corp);
    free_tree(state->tr);
    if (state->score_log != NULL)
        fclose(state->score_log);
    if (state->tree_structure_log != NULL)
        fclose(state->tree_structure_log);
    if (state->run_dir != NULL)
        free(state->run_dir);
    free(state);
}
