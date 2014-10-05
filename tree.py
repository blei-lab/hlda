#! /usr/bin/python

# how to use this:
#
# docmap = tree.read_jacm_docmap("/Users/blei/data/jacm/current-jacm/jacm-doc.map")
# state = tree.read_state("GOO/mode", vocab, 5)
# tree.add_assignments_to_tree('GOO/mode.assign', state['tree'])
# tree.write_topic_tree_ascii(state, docmap, "GOO.txt")
# tree.write_topic_tree_dot(goo, "GOO.dot", 0, 0)

import sys, re, os, itertools, math

VOCAB = '/Users/blei/data/jacm/002/jacm-vocab.dat'
DOCMAP = '/Users/blei/data/jacm/003/jacm-doc.map'

def doc_sort_key(x, docmap, level):
    return(-(x[1] + math.log(docmap[x[0]]['counts'].get(level,1e-5))))


def top_n_words(topic,
                vocab,
                nwords):
    """
    the top n words from a topic
    vocab is a map from integers to words

    """
    indices = range(len(vocab))
    indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
    return([vocab[i] for i in indices[0:nwords]])


def compute_level(id, tree):
    """
    compute the level of an id in a tree

    """
    topic = tree[id]
    level = 0
    while (id != 0):
        level += 1
        id = topic['parent']
        topic = tree[id]
    return(level)


def read_state(state_filename,
               vocab,
               sig_size):
    """
    read the state from an iteration file (e.g., mode)

    """
    state = file(state_filename, 'r')

    score         = float(state.readline().split()[1])
    iter          = int(state.readline().split()[1])
    eta           = state.readline().split()
    eta           = [float(x) for x in eta[1:len(eta)]]
    gam           = state.readline().split()
    gam           = [float(x) for x in gam[1:len(gam)]]
    gem_mean      = float(state.readline().split()[1])
    gem_scale     = float(state.readline().split()[1])
    scaling_shape = float(state.readline().split()[1])
    scaling_scale = float(state.readline().split()[1])

    header = state.readline()
    tree = {}
    for line in state:

        (id, parent, ndocs, nwords, scale, word_cnt) = line.split(None, 5)
        (id, parent, ndocs, nwords) = [int(x) for
                                       x in [id, parent, ndocs, nwords]]
        scale = float(scale)
        tree[id] = {}
        tree[id]['parent'] = parent
        if (parent >= 0): tree[parent]['children'].append(id)

        tree[id]['nwords'] = nwords
        tree[id]['ndocs'] = ndocs
        tree[id]['scale'] = scale
        topic = [int(x) for x in word_cnt.split()]
        tree[id]['top_words'] = top_n_words(topic, vocab, sig_size)
        tree[id]['children'] = []

    for topic in tree.values():
        topic['children'].sort(key=lambda id: -tree[id]['ndocs'])

    return({'score':score,
            'iter':iter,
            'gam':gam,
            'eta':eta,
            'gem_mean':gem_mean,
            'gem_scale':gem_scale,
            'scaling_shape':scaling_shape,
            'scaling_scale':scaling_scale,
            'tree':tree})


def add_assignments_to_tree(filename, tree):
    """
    reads an iter.assign file and adds document IDs to the leaf
    topics.  with a doc-map, we can associate titles with topics

    """
    for line in file(filename, 'r'):
        (doc_id, score, path) = line.split(None, 2)
        doc_id = int(doc_id)
        score = float(score)
        path = [int(x) for x in path.split()]
        for topic in path:
            tree[topic].setdefault('docs', []).append((doc_id, score))



def add_state_to_dmap(name, vocab, dmap):
    """
    read the level assignments '<ITER>.levels'
    and the topic assignments '<ITER>.assign'
    and adds them to the document map

    (note: vocab is a mapping from numbers to vocabulary words)

    """
    dnum = 0
    for (levels, topics) in itertools.izip(file(name+'.levels'),
                                           file(name+'.assign')):

         (id, score, path) = topics.split(None, 2)
         id = int(id)
         dmap[id]['score'] = float(score)
         dmap[id]['path'] = [int(c) for c in path.split()]

         zvars = {}
         counts = {}
         items = levels.split()
         for item in items:
             (word, level) = [int(x) for x in item.split(':')]
             counts[level] = counts.get(level, 0) + 1
             zvars.setdefault(vocab[word], []).append(level)
         dmap[id]['levels'] = zvars
         dmap[id]['counts'] = counts


def read_vocab_map(vocab_file):
    """
    given a vocabulary file, returns a mapping from integers to words.

    """
    num = 0
    vocab = {}
    for word in file(vocab_file):
        vocab[num] = word.strip()
        num = num + 1
    return(vocab)


def read_jacm_docmap(filename):
    """
    read the jacm doc-map, which includes the title and abstract

    """
    docs = {}
    doc_id = 0
    for line in file(filename, 'r'):
        (bad_doc_id, title, abstract) = [x.replace('"', '') for x in line.split(' "')]
        nwords = len(abstract.split())
        docs[doc_id] = {'title':title, 'abstract':abstract, 'nwords':nwords}
        doc_id += 1
    return(docs)


def read_docmap(filename):
    """
    read a doc-map, which is assumed to be a list of titles

    """
    docs = {}
    doc_id = 0
    for line in file(filename, 'r'):
        docs[doc_id] = {'title':line}
        doc_id += 1
    return(docs)


def write_docs(docs, tree, outfile):
    """
    writes a file with all the doc information

    """
    def word_and_level(word, level):
        return('%s_%d' % (word, level))

    out = file(outfile, 'w')
    for doc in docs:
        out.write(doc['title'] + '|')

        for topic in doc['path']:
            out.write(','.join(tree[topic]['top_words'])+'|')

        abstract = ' '.join([word_and_level(w, doc['levels'].get(w,[-1])[0])
                             for w in doc['abstract'].split()])
        out.write(abstract+'\n')
    out.close()

# write the topic tree with documents

def write_topic_tree_ascii(state,
                           docmap,
                           out_filename,
                           ndocs = -1,
                           min_ndocs = 0,
                           include_docs = False):

    out = file(out_filename, 'w')
    tree = state['tree']

    eta = ' '.join(['%1.3e' % x for x in state['eta']])
    gam = ' '.join(['%1.3e' % x for x in state['gam']])

    out.write('SCORE         = %s\n' % str(state['score']))
    out.write('ITER          = %s\n' % str(state['iter']))
    out.write('ETA           = %s\n' % eta)
    out.write('GAM           = %s\n' % gam)
    out.write('GEM_MEAN      = %s\n' % str(state['gem_mean']))
    out.write('GEM_SCALE     = %s\n' % str(state['gem_scale']))
    out.write('SCALING_SHAPE = %s\n' % str(state['scaling_shape']))
    out.write('SCALING_SCALE = %s\n\n' % str(state['scaling_scale']))

    max_level = len(state['gam'])

    def write_topic(topic, level):

        indent = '     ' * level
        out.write('%s' % indent)
        out.write("[%d/%d/%d]" % (level, topic['nwords'], topic['ndocs']))
        # out.write(' %s' % str(topic['scale']))
        out.write(' %s\n\n' % ' '.join([x.upper() for x in topic['top_words']]))

        if ((level == max_level) and include_docs):
        # if ((level > 0) and include_docs):
            docs = topic['docs']
            if (docmap[0].has_key('counts')):
                docs.sort(key=lambda x: doc_sort_key(x, docmap, level))
            if (ndocs > -1): docs = docs[0:ndocs]
            for (doc, score) in docs:
                # !!! this is broken if we don't have the counts
                out.write('%s %3.2f %s\n' %
                          (indent, doc_sort_key([doc,score], docmap, level),
                           docmap[doc]['title']))
                # out.write('%s %3.2f %s\n' %
                # (indent, score, docmap[doc]['title']))
        if (level == max_level): out.write('\n')
        for id in topic['children']:
            if ((tree[id]['ndocs'] >= min_ndocs) and
                (tree[id]['nwords'] > 0)):
                write_topic(tree[id], level + 1)

    write_topic(tree[0], 0)
    out.close()


# write the topic tree

def write_topic_tree_dot(state,
                         docmap,
                         out_filename,
                         min_ndocs = 2,
                         ndocs = -1,
                         join_char='\\n',
                         include_stats=False,
                         include_docs=True):

    outfile = file(out_filename, 'w')
    outfile.write("digraph topic_tree {\n")
    outfile.write("node [shape=egg, fontname=Helvetica];\n")
    outfile.write("edge [style=bold, arrowhead=dot, arrowsize=1];\n")
    outfile.write("graph [mindist=0];\n")

    eta       = ' '.join(['%1.3e' % x for x in state['eta']])
    gamma     = ' '.join(['%1.3e' % x for x in state['gam']])
    gem_mean  = str(state['gem_mean'])
    gem_scale = str(state['gem_scale'])
    score     = str(state['score'])
    iter      = str(state['iter'])

    max_level = len(state['gam'])

    outfile.write('params [shape=rectangle, style=bold, color=red,fontcolor=red, fontsize=24, label="ETA = %s\\nGAMMA = %s\\nGEM MEAN = %s\\nGEM SCALE=%s\\nSCORE = %s"]\n' %
                  (eta, gamma, gem_mean, gem_scale,score))

    skip = {}
    fontsizes = [24, 18, 12, 9]

    id = 0

    def write_topic(topic, id, level):

        label = join_char.join(topic['top_words'])
        if include_stats:
            label = '[%d/%d]\\n' % (topic['nwords'],topic['ndocs']) + label

        outfile.write('%d [fontsize=%s, label="%s"];\n' %
                      (id, fontsizes[level], label))
        outfile.write('%d -> %d;\n' % (topic['parent'], id))

        if ((level == max_level) and include_docs):
            docs = topic['docs']
            docs.sort(key=lambda x: doc_sort_key(x, docmap, level))
            if (ndocs > -1): docs = docs[0:ndocs]
            docs_label = join_char.join([docmap[doc[0]]['title']
                                         for doc in docs])
            docs_id = '%d' % (id * 10 + 1)
            outfile.write('%s [fontsize=9, label="%s"];\n' %
                          (docs_id, docs_label))
            outfile.write('%d -> %s;\n' % (id, docs_id))

        children = sorted(topic['children'], key=lambda x: -tree[x]['ndocs'])
        for id in children:
            if (tree[id]['ndocs'] >= min_ndocs):
                write_topic(tree[id], id, level + 1)

    tree = state['tree']
    write_topic(tree[0], 0, 0)
    outfile.write("}")
    outfile.close()


# walk down a directory and make both text and dot trees using a
# single vocabulary and dmap.

# tree.make_all_trees('fits/DP-nested/jacm/006/', 'data/jacm/002/jacm-vocab.dat', 'data/jacm/003/jacm-doc.map')

def make_all_trees(dir,
                   vocab_filename,
                   dmap_filename,
                   sig_size=10,
                   ndocs=-1,
                   home=os.environ['HOME']):

    vocab = map(str.strip, file(home+'/'+vocab_filename, 'r').readlines())
    # docmap = read_docmap(dmap_filename)
    docmap = read_jacm_docmap(home+'/'+dmap_filename)

    walk = os.walk(home+'/'+dir)
    max_score = None
    argmax_dir = None
    for dir, _, files in walk:
        files = filter(lambda x: x=='mode', files)
        for f in files:
            sys.stderr.write('WRITING %s/%s\n' % (dir, f))
            filename = dir+'/'+f
            state = read_state(filename, vocab, sig_size)
            if (state['score'] > max_score):
                max_score = state['score']
                argmax_dir = dir
            add_assignments_to_tree(filename+'.assign', state['tree'])
            add_state_to_dmap(filename, vocab, docmap)
            txt_tree = dir+'/mode.txt'
            dot_tree = dir+'/mode.dot'
            write_topic_tree_ascii(state, docmap, txt_tree, ndocs=ndocs)
            write_topic_tree_dot(state, docmap, dot_tree, ndocs=10)

    sys.stderr.write("BEST RUN = %s\n" % argmax_dir)

# main function

def main(type,
         iter_filename,
         vocab_filename,
         dmap_filename,
         out_filename,
         sig_size = 5,
         ndocs = -1):

    vocab = map(str.strip, file(vocab_filename, "r").readlines())
    state = read_state(iter_filename, vocab, sig_size)
    add_assignments_to_tree(iter_filename+'.assign', state['tree'])

    if (type == 'txt'):
        if os.path. isfile(dmap_filename):
            docmap = read_docmap(dmap_filename)
        write_topic_tree_ascii(state, docmap, out_filename, ndocs=ndocs)
    else:
        write_topic_tree_dot(state, out_filename)


if (__name__ == '__main__'):

    if (len(sys.argv) != 6):
        sys.stdout.write('usage: python tree.py <txt/dot> <iter> <vocab> <dmap> <out>\n')
        sys.exit(1)

    type = sys.argv[1]
    iter_filename = sys.argv[2]
    vocab_filename = sys.argv[3]
    dmap_filename = sys.argv[4]
    out_filename = sys.argv[5]

    main(type, iter_filename, vocab_filename, dmap_filename, out_filename)
