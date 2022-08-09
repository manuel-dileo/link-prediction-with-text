from gensim.parsing.preprocessing import preprocess_documents
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pandas as pd
import numpy as np
import math
from math import log
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
from collections import Counter

def process_text(extracted):
    """
    Given a list of comment_op extracted from blockchain it returns 
        - the entire corpus of the snap,
        - the corpus related to post, the corpus related to comment, the corpus related to tags,
        - a dictionary in which for each author is stored its entire corpus, its used tags, its comment, its post
        - the list of all tags.
    To understand better the data fields see the Steemit API doc.
    Text-based statistics can be easily obtained after this step.
    """
    authors = {}
    processed = 0
    end = False
    tags = []
    entire_corpus = []
    corpus_post = []
    corpus_comment = []
    corpus_tag = []
    num_post = 0
    num_comment = 0
    num_tag_per_doc = 0
    while not end:
        try:
            raw = next(extracted)
        except StopIteration:
            end = True
        
        comm = raw['value']
        #data fields are encrypted in our storage so we need to decrypt them
        parent_author = chain_reader.decrypt(comm['parent_author'])
        author = chain_reader.decrypt(comm['author'])
        body = chain_reader.decrypt(comm['body'])
        
        if author not in authors:
            authors[author] = {'corpus': [], 'tags': [], 'corpusPost': [], 'corpusComment': [], 'corpusTag': []}
            
        
        try:
            meta = json.loads(chain_reader.decrypt(comm['json_metadata'])) #read metadata to extract tags
        except json.JSONDecodeError: 
            meta = ''
            
        try:
            if 'tags' in meta:
                tag = meta['tags']
                authors[author]['corpus'].append(' '.join(tag))
                entire_corpus.append(' '.join(tag))
                authors[author]['tags'].extend(tag)
                authors[author]['corpusTag'].append(' '.join(tag))
                tags.extend(tag)
                corpus_tag.append(' '.join(tag))
                num_tag_per_doc += len(tag)
        except TypeError: pass
        
        authors[author]['corpus'].append(body)
        entire_corpus.append(body)
        if parent_author == '':
            authors[author]['corpusPost'].append(body)
            corpus_post.append(body)
            num_post += 1
        else:
            authors[author]['corpusComment'].append(body)
            corpus_comment.append(body)
            num_comment+=1
        
        processed += 1
        if processed % 100000 == 0:
            print("processed ", processed, " comment op")
    print("comment op ", processed)
    avg_tag = num_tag_per_doc / (num_post + num_comment)
    return entire_corpus, corpus_post, corpus_comment, corpus_tag, authors, tags, num_post, num_comment, avg_tag

def get_author_lda(ldaLabel,entire_corpus,authors,K,aggregator=np.mean):
    """
        Get user interest vector based on LDA distributions over its content.
        The entire collection of content is preprocessed through some Gensim filters
        An LDA model is fed with the entire collection of content posted on Steemit in a certain time period.
        The trained LDA model returns for each user for each content a probability distribution over a certain number of topics K.
        To obtain the user interest vector we aggregate the topic distributions over its content (for instance, by averaging them)
        ldaLabel allow to specify if you want an user interest vector based on the entire corpus, only the posts, comments or tags.
        You can easily compute distances between user interest vectors after this step
    """
    if ldaLabel == 'entire':
        corpusLabel = 'corpus'
    elif ldaLabel == 'post':
        corpusLabel = 'corpusPost'
    elif ldaLabel == 'comment':
        corpusLabel = 'corpusComment'
    else:
        corpusLabel = 'corpusTag'
    
    author_topics = {}
    entire_prep_corpus = preprocess_documents(entire_corpus)
    dct = Dictionary(entire_prep_corpus)
    bow_entire_corpus = [dct.doc2bow(text) for text in entire_prep_corpus]
    lda = LdaModel(bow_entire_corpus,num_topics=K)
    
    
    for author, c in authors.items():
        rawcorpus = c[corpusLabel]
        preprocess_corpus =  preprocess_documents(rawcorpus)
        dct = Dictionary(preprocess_corpus)
        corpus = [dct.doc2bow(text) for text in preprocess_corpus]
        #flat_corpus = [item for sublist in corpus for item in sublist]
        #doc can be empty (meaningless) after preprocessing
        #in this way we obtain a probability distribution over topic
        topic_distribs = dict([(i,[0]) for i in range(K)]) 
        for doc in corpus: 
            vec = lda[doc]
            for j,v in vec:
                topic_distrib = topic_distribs[j]
                if len(topic_distrib) == 1 and topic_distrib[0] == 0: #drop the initial zero
                    topic_distribs[j] = []
                topic_distribs[j].append(v)
    
            author_topics[author] = dict([(k,aggregator(v)) for k,v in topic_distribs.items()])
        
    return author_topics

