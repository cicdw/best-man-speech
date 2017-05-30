import html.parser as htmlparser
import gensim
import mailbox
import numpy as np
import pickle
import spacy
from gensim import corpora


nlp = spacy.load('en')


class Email(object):

    @staticmethod
    def _get_mailbox(fname):
        '''Reads mailbox from .mbox or pickle file'''
        if fname.endswith('mbox'):
            out = mailbox.mbox(fname)
        else:
            with open(fname, 'rb') as fp:
                out = pickle.load(fp)
        return out

    def filter_by_email(self, by=None, to=None):
        out = []
        for msg in self.mailbox:
            keep = True
            if by is not None:
                who = msg['From']
                if isinstance(who, str):
                    keep = by.lower() in who.lower()
            if to is not None:
                who = msg.get('To', '')
                if isinstance(who, str):
                    keep = keep or (to.lower() in who.lower())
            if keep:
                out.append(msg)
        return Email(out)

    @property
    def content(self):
        if not hasattr(self, '_content'):
            self._content = self._get_content()
        return self._content

    def _get_content(self):
        '''Converts current mailbox object to a list of strings
        representing the content of each email.'''
        msgs = []
        for msg in self.mailbox:
            content = msg.get_payload()
            while isinstance(content, list):
                content = content[0].get_payload()
            msgs.append(content.lower())
        return msgs

    def save(self, fname):
        with open(fname, 'wb') as fp:
            pickle.dump(self.mailbox, fp)

    @classmethod
    def from_file(cls, fname):
        '''Creates Email object from file.'''
        mailbox = cls._get_mailbox(fname)
        return cls(mailbox)

    def __init__(self, mailbox):
        self.mailbox = mailbox


class Worker(object):

    def _keep_condition(self, word):
        keep = not word.is_stop
        keep = keep and not word.is_punct
        keep = keep and not word.is_space
        keep = keep and not any(bad in word.lemma_.lower() for bad in self.stoppers)
        keep = keep and len(word) > 2
        return keep

    def _replace_word(self, word):
        for inputs, outputs in self.replacements.items():
            if inputs in outputs:
                return outputs

    def clean_msg(self, msg):
        out = []
        for w in nlp(msg):
            if self._keep_condition(w):
                tmp = htmlparser.unescape(w.lemma_.lower())
                out.append(self._replace_word(tmp))
        return out

    def _list_agg(self, a, n=1):
        out = []
        for elems in zip(*[a[s::n] for s in range(n)]):
            out.append([item for sublist in elems for item in sublist])
        if (len(a) % n) != 0:
            num_app = -(len(a) - n) if (2*n > len(a)) else -n + 1
            out.append([item for sublist in a[num_app:] for item in sublist])
        return out

    def tokenize_msgs(self, msgs, min_length=1, keep_n=3000, no_above=0.5, num_merge=0):
        tokens = [self.clean_msg(msg) for msg in msgs]
        if num_merge > 0:
            tokens = self._list_agg(tokens, n=num_merge + 1)
        self.dictionary = corpora.Dictionary(tokens)
        self.dictionary.filter_extremes(no_above=no_above, keep_n=keep_n)
        self.corpus = [self.dictionary.doc2bow(msg) for msg in tokens]
        self.corpus = [msg for msg in self.corpus if len(msg) >= min_length]
        self._id2word = {idx: tkn for tkn, idx in self.dictionary.token2id.items()}
        return tokens

    def __getitem__(self, key):
        if isinstance(key, (int, np.int64)):
            return self._id2word[key]
        else:
            return self.dictionary.token2id.get(key, None)

    def __len__(self):
        return len(self.dictionary.token2id)

    @property
    def word_counts(self):
        return self.word_count_matrix.sum(axis=1)

    @property
    def word_count_matrix(self):
        if hasattr(self, '_matrix'):
            return self._matrix
        else:
            num_tokens = len(self.dictionary.token2id)
            matrix = gensim.matutils.corpus2dense(self.corpus,
                                                  num_terms=num_tokens)
            self._matrix = matrix
            return matrix

    def __init__(self, stoppers=None, replacements=None):
        self.replacements = replacements
        self.stoppers = stoppers
