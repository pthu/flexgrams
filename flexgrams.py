
# coding: utf-8

# In[ ]:


#===================================================#
# This program applies intertextual phrase matching #
# to one or more files in the following formats:    #
#   - plain python strings, tuples, lists           #
#   - plain text files                              #
#   - csv-files (TODO)                              #
#   - TEI XML files (internally converted to TF)    #
#   - Text-Fabric file packages                     #
#                                                   #
# Author: Ernst Boogert                             #
# Institution: Protestant Theological University    #
# Place: Amsterdam, the Netherlands                 #
#                                                   #
# Version: 0.0.7 (stable version)                   #
# Last modified: May 29th, 2019                     #
#===================================================#

import os
from pprint import pprint
from pandas import DataFrame
from collections import defaultdict, OrderedDict
from tf.fabric import Fabric, Timestamp
from collatex import Collation, collate
from itertools import combinations_with_replacement as selfcombinations
from itertools import combinations, product

get_ipython().run_line_magic('run', './helpertools/tokenizer.ipynb')

tm = Timestamp()

class FlexGrams:
    def __init__(self, base_path, comp_path=None, ngram=5, 
                 skip=1, ngram_type="ordered", start=0, stop=0, steps=1,
                 mode=0, distance_base=0, distance_comp=0, 
                 number=1, context=0, sort_order="base", self_match=False,
                 stopwords=None, synonyms=None):
        # External variables
        self.base_path     = base_path
        self.comp_path     = comp_path
        self.ngram         = ngram
        self.skip          = skip
        self.ngram_type    = ngram_type
        self.start         = start
        self.stop          = stop
        self.steps         = steps
        self.mode          = mode
        self.distance_base = distance_base
        self.distance_comp = distance_comp
        self.number        = number
        self.context       = context
        self.sort_order    = sort_order
        self.self_match    = self_match
        self.stopwords     = stopwords         # Should be a set
        self.synonyms      = synonyms          # Should be a dict
        
        # Internal Variables
        self.base_sources  = OrderedDict()
        self.comp_sources  = OrderedDict()
        self.tf_apis_dict  = {}
        self.source_count  = 0
        self.result_dict   = self.ipm()
        
    
    def path_loader(self, path):
        sources_dict = {}
        
        # Helper functions that return action upon path
        def walk_file(file):
            name, ext = os.path.splitext(file)
            if ext == '.xml':
                pass # TODO: apply TEI xml converter
            elif ext == '.txt':
                                                     #self.non_tf_mode(open(file, 'r').read())
                sources_dict[f'{self.source_count}_{name}'] = [word for word in open(file, 'r').read().split()]
                self.source_count +=1
            elif ext == '':
                try:
                    sources_dict[f'{self.source_count}_{name}'] = [word for word in open(file, 'r').read().split()]
                    self.source_count +=1
                except:
                    print("No valid input...!")
                        
        def walk_dir(path):
            if os.path.isdir(path):
                if os.path.isfile(path + '/oslots.tf'):
                    self.tf_apis_dict[f'TF_{self.source_count}'] = self.tf_initiate(path)
                    sources_dict[f'TF_{self.source_count}'] = self.tf_mode(self.tf_apis_dict[f'TF_{self.source_count}'])
                    self.source_count +=1
                else:
                    with os.scandir(path) as it:
                        for entry in it:
                            if entry.is_dir():
                                walk_dir(entry.path)
                            elif entry.is_file():
                                walk_file(entry.path)
                            else:
                                print('Something went wrong while walking the dir...')

        if os.path.isfile(path):              # If it is a single file: try to read and to tokenize it
            walk_file(path) 
        elif os.path.isdir(path):             # If it is a dir: scan the dirs and files inside
            walk_dir(path)  
        elif isinstance(path, str):           # If it is a string: tokenize the string
            sources_dict[f'{self.source_count}_string'] = [path.split() if ' ' in path else path]
            self.source_count +=1
        elif isinstance(path, (list, tuple)): # If it is a tuple or list: pass it on to the algoritm
            sources_dict[f'{self.source_count}_list'] = [path]
            self.source_count +=1
        else:
            print('Input error: are you sure that your input is valid?')
        tm.info(f'{self.source_count} source text(s) found...')
        return sources_dict
        
    
    def non_tf_mode(self, raw):
        ''' Explanation modes:
        In all modes, standard normalization will be applied...
        0 = exact comparison
        1 = unaccented exact comparison
        2 = accented lemmatized comparison
        3 = unaccented lemmatized comparison
        4 = betacode accented comparison
        5 = betacode unaccented comparison
        '''
        modes = {
                0: tokenize(raw),
                1: None,
        }
    
    def tf_mode(self, api): #TODO!
        ''' Explanation modes:
        In all modes, standard normalization will be applied...
        0 = exact comparison
        1 = unaccented exact comparison
        2 = accented lemmatized comparison
        3 = unaccented lemmatized comparison
        4 = betacode accented comparison
        5 = betacode unaccented comparison
        '''
        if self.stop == 0:
            stop = api.F.otype.s('word')[-1] #api.F.otype.maxSlot
        else:
            stop = self.stop
        if self.start == 0:
            start = api.F.otype.s('word')[0]
        # The format of the list comprehensions is necessary to ensure that for each slot something is returned, 
        # even if it is empty, because otherwise slot numbers will not correspond with the internal numbers
        modes = { 
                0: [api.T.text(token, fmt='text-orig-main') for token in range(start, stop)], # normalized original without punctuation
                1: [api.T.text(token, fmt='text-orig-full') for token in range(start, stop)], # original form 
                2: [api.T.text(token, fmt='text-orig-plain') for token in range(start, stop)], #tf main-regularized-unicode-unaccented
                3: [api.T.text(token, fmt='text-orig-lemma') for token in range(start, stop)], #tf main-lemmatized-accented
                4: [],#tf main-lemmatized-unaccented
                5: [api.T.text(token, fmt='text-orig-beta') for token in range(start, stop)],#tf main-betacode-accented
                6: [api.T.text(token, fmt='text-orig-beta-plain') for token in range(start, stop)],#tf main-betacode-unaccented
        }
#         print(modes[self.mode])
        return modes[self.mode] # returns a list of words, to be fed into the ngrams_generator

    
    def ngrams_generator(self, source, text):
        '''The ngrams_generator is a modified version of the
        ITERTOOLS: combinations() function.'''
        # Setting up the parameters
        ngram = self.ngram
        skip = self.skip
        if self.ngram_type == "ordered":
            ngram_type = tuple
        elif self.ngram_type == "unordered":
            ngram_type = frozenset
        start = self.start
        if self.stop == 0:
            stop = len(text)
        else:
            stop = self.stop
        steps = self.steps

        #Doing the loop and output ngrams in format (startindex, (word1, word2, word3, ...))
        while start <= stop - ngram:
            ngrams = text[(start):(start + ngram)]
            length_ngrams = len(ngrams)                     # Calculate the length of the ngram_pool
            result = ngram - skip                           # Calculate the length of the result
            indices = list(range(result))                   # Creates the initial index to calculate the startpoint of the resulting ngrams
            yield (start, ngram_type(ngrams[i] for i in indices))

            while True:
                for i in reversed(range(result)):
                    if indices[i] != i + length_ngrams - result:  # Checks whether the result is not equal to the initial resulting ngram
                        break
                indices[i] +=1                           # Updates all index items by one
                for j in range(i+1, result):
                    indices[j] = indices[j-1] + 1  
                if indices[0] > 0:                       # If the skip becomes the first element, the algorithm goes to the next ngram to prevent duplicates
                    start += steps
                    break
                yield (start, ngram_type(ngrams[i] for i in indices))

    
    def ngrams_dict(self, sources_dict):
        ngrams_dict = OrderedDict()
        for source, text in sources_dict.items():
            ngrams_dict[source] = defaultdict(list)
            for (start, ngram) in self.ngrams_generator(source, text):
                ngrams_dict[source][ngram].append(start)
#         pprint(ngrams_dict)
        return ngrams_dict

    
    def matcher(self, base_ngrams_dict, comp_ngrams_dict=None):
        '''The matcher function returns a dict
        with matched sources as keys in a tuple: (source1, source2)
        and the matching ngram slices as list of tuples: [(1, 5), (8, 16), ...]'''
        #TODO: implement fuzzywuzzy as alternative for skips, or implement set methods
        match_dict = {} #OrderedDict() #defaultdict(list)
                
        if comp_ngrams_dict == None: # If this is the case, the algorithm conducts a selfcomparison...
            if len(base_ngrams_dict) == 1: # In this case, the algorithm expects a selfmatch, even is self.self_match is false (=default)
                key = next(iter(base_ngrams_dict))
                match_dict[(key, key)] = sorted([ (i, i+self.ngram, j, j+self.ngram, 1) for v in base_ngrams_dict[key].values()                                                          for i in v[0:len(v)-1]                                                          for j in v[1:len(v)] if j > i + self.ngram ])
            else: #base_ngrams_dict contains more than one source
                if self.self_match == False:
                    dictloop = combinations(base_ngrams_dict.keys(), 2)
                else:
                    dictloop = selfcombinations(base_ngrams_dict.keys(), 2) # Returns all source combinations including selfmatch
                for loop1, loop2 in dictloop:
                    if loop1 == loop2: # = selfmatch!
                        match_dict[(loop1, loop2)] = sorted([ (i, i+self.ngram, j, j+self.ngram, 1) for v in base_ngrams_dict[loop1].values()                                                               for i in v[0:len(v)-1]                                                               for j in v[1:len(v)]                                                               if j > i + self.ngram ])
                    else:
                        match_dict[(loop1, loop2)] = sorted([ (i, i+self.ngram, j, j+self.ngram, 1) for k1, v1 in base_ngrams_dict[loop1].items()                                                               for i in v1                                                               for j in base_ngrams_dict[loop2][k1]                                                               if k1 in base_ngrams_dict[loop2] ])
                                                              
        else:
            dictloop = product(base_ngrams_dict.keys(), comp_ngrams_dict.keys())
            for loop1, loop2 in dictloop:
                match_dict[(loop1, loop2)] = sorted([ (ib, ib+self.ngram, ic, ic+self.ngram, 1) for kc, vc in comp_ngrams_dict[loop2].items()                                                                for ic in vc                                                                for ib in base_ngrams_dict[loop1][kc]                                                                if kc in base_ngrams_dict[loop1]])
            
#         pprint(match_dict)
        return match_dict
        

    def concatenator(self, match_dict):
        
        def worker(match_list):
            if match_list != []:
                cmatch_list = [match_list[0]]
                for i in match_list[1:]:
                    j = 1
                    assigned = False
                    # Check whether the base slice fits in the allowed range
                    # Since the list is sorted, i[0] is always greater than the one before
                    # First, we check whether the base start index number (=i[0]) fits within the allowed range...
                    while i[0] <= cmatch_list[-j][1] + self.distance_base:
                        # Then we check whether the comp start index number (=i[2]) fits with in the allowed range
                        if i[2] >= cmatch_list[-j][2] - self.distance_comp - 1 and                            i[2] <= cmatch_list[-j][3] + self.distance_comp + 1:
                            # If both checks are True, we process the concatenation...
                            if i[2] >= cmatch_list[-j][2]:
                                if i[3] > cmatch_list[-j][3]:
                                    cmatch_list[-j] = tuple((cmatch_list[-j][0],
                                                             i[1],
                                                             cmatch_list[-j][2],
                                                             i[3],
                                                             cmatch_list[-j][4] + i[4] if not cmatch_list[-j][1] == i[1] else cmatch_list[-j][4], 
                                                             ))
                                    assigned = True
                                    break
                                else:
                                    cmatch_list[-j] = tuple((cmatch_list[-j][0],
                                                             i[1],
                                                             cmatch_list[-j][2],
                                                             cmatch_list[-j][3],
                                                             cmatch_list[-j][4], # + i[4] if not cmatch_list[-j][1] == i[1] else cmatch_list[-j][4], 
                                                             ))
                                    assigned = True
                                    break
                             
                            else: # i[2] <= cmatch_list[-j][2]]
                                if i[3] < cmatch_list[-j][3]:
                                    cmatch_list[-j] = tuple((cmatch_list[-j][0],
                                                             i[1],
                                                             i[2],
                                                             cmatch_list[-j][3],
                                                             cmatch_list[-j][4] + i[4] if not cmatch_list[-j][1] == i[1] else cmatch_list[-j][4], 
                                                             ))
                                    assigned = True
                                    break
                                else: #i [3] >= cmatch_list[-j][3]
                                    cmatch_list[-j] = tuple((cmatch_list[-j][0],
                                                             i[1],
                                                             i[2],
                                                             i[3],
                                                             cmatch_list[-j][4] + i[4] if not cmatch_list[-j][1] == i[1] else cmatch_list[-j][4], 
                                                             ))
                                assigned = True
                                break
                            
                        # If the comp slice does not fit: 
                        else:
                            # Check one slice earlier
                            j +=1
                            if j > len(cmatch_list):
                                if assigned == False:
                                    cmatch_list.append(i)
                                break
                            continue
                    else:
                        if assigned == False:
                            cmatch_list.append(i)
#                 pprint(cmatch_list)
            else:
                cmatch_list = []
                
            if len(cmatch_list) != len(match_list):
                return worker(cmatch_list)
            else:
                return cmatch_list
#                 fcmatch_dict = {key: [tuple((i[0], i[1], i[2], i[3], i[0]-self.ngram)) for i in value] for key, value in cmatch_dict.items()}
        for sm in match_dict:
            match_dict[sm] = worker(match_dict[sm])
        fmatch_dict = {}
        for sm in match_dict:
            fmatch_dict[sm] = [s for s in match_dict[sm] if s[4] >= self.number]
#         pprint(fmatch_dict)
        return fmatch_dict

    def context_concatenator(self):
        '''Concatenates matches that fall within the context parameter'''
        # Not finished!!!
        contextualized_match_dict = {}
        return contextualized_match_dict
        
        
    def ipm(self):
        # Load all tokenized packages in base_path in base_sources dictionary
        # Send base_path to path_loader --> path_loader() uses mode_tokenizer() to decide how to tokenize
        self.base_sources = self.path_loader(self.base_path) #Get a dict with one or more base texts
        base_ngrams_dict = self.ngrams_dict(self.base_sources)
        if self.comp_path == None:
            match_dict = self.matcher(base_ngrams_dict)
            fcmatch_dict = self.concatenator(match_dict)

        # Load all tokenized packages in comp_path one by one into memory,
        # execute comparison and delete them from memory subsequently
        # Walk through comp_path and send them one by one to the path_loader()
        # Path(loader) returns a dict with the name of the package (=key) and a tokenized list (=value)
        else:
            #Execute comparison between base and comp
#           for file in self.comp_path: # TODO!!! Convert to os.path expression and make them load one by one
            self.comp_sources = self.path_loader(self.comp_path)
            comp_ngrams_dict = self.ngrams_dict(self.comp_sources)
            match_dict = self.matcher(base_ngrams_dict, comp_ngrams_dict)
            fcmatch_dict = self.concatenator(match_dict)
#         pprint(fcmatch_dict)
        return fcmatch_dict

    def getRefList(self, api, levels='all'):
        nodeType = api.TF.features['otext'].metaData['availableStructure'].split(',')[-1]
        sectionNodes = api.F.otype.s(nodeType)
#         refList = [".".join(str(i[1]) for i in api.T.headingFromNode(node)[1:]) for node in sectionNodes]
        refs = [tuple(str(i[1]) for i in api.T.headingFromNode(node)[1:]) for node in sectionNodes]
        refList = list(OrderedDict.fromkeys(refs))
#         list(OrderedDict.fromkeys( self.getRefList(api_b, levels='all') ))
        if levels == 'all':
            return ['.'.join(ref) for ref in refList]
        else:
            return ['.'.join([ref[level] for level in levels]) for ref in refList]
            
    
    def refResult(self, order='base', b_levels='all', c_levels='all'):
        frame_dict = OrderedDict(bibl_start=[], bibl_stop=[],
                                 patr_start=[], patr_stop=[],
                                 typ=[], conf=[], 
                                 source=[], found=[],
                                 base_text=[], comp_text=[])
        result_dict = self.result_dict
        for sm in result_dict: # sm = source match = tuple of two documents that have matching ngrams
            if sm[0].startswith('TF_'):
                api_b = self.tf_apis_dict[sm[0]]
                b_min, b_max = api_b.F.otype.s('word')[0], api_b.F.otype.s('word')[-1]
            if sm[1].startswith('TF_'):
                api_c = self.tf_apis_dict[sm[1]]
                c_min, c_max = api_c.F.otype.s('word')[0], api_c.F.otype.s('word')[-1]
                
            for match in result_dict[sm]:    
                if sm[0].startswith('TF_'):
                    # Definition of references
                    b_start = [ str(i[1]) for i in api_b.T.headingFromNode(api_b.L.u(match[0]+1, otype=api_b.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    b_stop = [ str(i[1]) for i in api_b.T.headingFromNode(api_b.L.u(match[1], otype=api_b.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    c_start = [ str(i[1]) for i in api_c.T.headingFromNode(api_c.L.u(match[2]+1, otype=api_c.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    c_stop = [ str(i[1]) for i in api_c.T.headingFromNode(api_c.L.u(match[3], otype=api_c.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    
                    con = 5
                    base_start = match[0] - con if not match[0] - con < b_min else b_min
                    base_stop = match[1] + con if not match[1] + con > b_max else b_max
                    comp_start = match[2] - con if not match[2] - con < c_min else c_min
                    comp_stop = match[3] + con if not match[3] + con > c_max else c_max
                    
                frame_dict['bibl_start'].append('.'.join(b_start)) if b_levels == 'all'                     else frame_dict['bibl_start'].append( '.'.join([b_stop[level] for level in b_levels]) ) 
                frame_dict['bibl_stop'].append('.'.join(b_stop)) if b_levels == 'all'                     else frame_dict['bibl_stop'].append( '.'.join([b_start[level] for level in b_levels]) ) 
                frame_dict['patr_start'].append('.'.join(c_start)) if c_levels == 'all'                     else frame_dict['patr_start'].append( '.'.join([c_start[level] for level in c_levels]) ) 
                frame_dict['patr_stop'].append('.'.join(c_stop)) if c_levels == 'all'                     else frame_dict['patr_stop'].append( '.'.join([c_stop[level] for level in c_levels]) )
                frame_dict['typ'].append('undefined')    
                frame_dict['conf'].append('U')
                frame_dict['source'].append('Flexgrams')
                frame_dict['found'].append('False')
                frame_dict['base_text'].append( ' '.join([ token for token in api_b.T.text(range(base_start, base_stop), fmt='text-orig-full', descend=True).split() ]).strip() )
                frame_dict['comp_text'].append( ' '.join([ token for token in api_c.T.text(range(comp_start, comp_stop), fmt='text-orig-full', descend=True).split() ]).strip() )
                    
        data = DataFrame(frame_dict)
        
        # Build refLists
        refListBase = self.getRefList(api_b, levels='all') 
        refListComp = self.getRefList(api_c, levels=(0, 2, 3)) 
        sortIndexBase = dict(zip(refListBase, range(len(refListBase))))
        sortIndexComp = dict(zip(refListComp, range(len(refListComp))))
        
        # Add sort columns
        data['r_bibl_start'] = data['bibl_start'].map(sortIndexBase)
        data['r_patr_start'] = data['patr_start'].map(sortIndexComp)
        if order == 'base':
            data.sort_values(['r_bibl_start', 'r_patr_start', 'patr_stop', 'bibl_stop'],                               ascending = [True, True, True, True], inplace=True)
        else:
            data.sort_values(['r_patr_start', 'r_bibl_start', 'patr_stop', 'bibl_stop'],                               ascending = [True, True, True, True], inplace=True)
        # Delete sort columns
        data.drop('r_bibl_start', 1, inplace=True)
        data.drop('r_patr_start', 1, inplace=True)
        
#         print(data)
        return data, refListBase, refListComp
        
                        
    def allign(self): #fcmatch_dict is to be changed to the contextualized_match_dict
        fcmatch_dict = self.result_dict
        collation = Collation()
        result_list = []
        for sm in fcmatch_dict: # sm = source match = tuple of two documents that have matching ngrams
            if sm[0].startswith('TF_'):
                api_b = self.tf_apis_dict[sm[0]]
                b_min, b_max = api_b.F.otype.s('word')[0], api_b.F.otype.s('word')[-1]
            else:
                b_min, b_max = 0, len(list(self.base_sources[sm[0]]))
               
            if sm[1].startswith('TF_'):
                api_c = self.tf_apis_dict[sm[1]]
                c_min, c_max = api_c.F.otype.s('word')[0], api_c.F.otype.s('word')[-1]
            else:
                c_min, c_max = 0, len(list(self.comp_sources[sm[1]])) if not self.comp_path == None else len(list(self.base_sources[sm[1]]))
            
            for match in fcmatch_dict[sm]:
#                 print(match)
                base_start = (match[0] - self.context) if not (match[0] - self.context) < b_min else b_min
                base_stop  = (match[1] + self.context) if not (match[1] + self.context) > b_max else b_max # Make mechanism to measure max length and to test it...
                comp_start = (match[2] - self.context) if not (match[2] - self.context) < c_min else c_min
                comp_stop  = (match[3] + self.context) if not (match[3] + self.context) > c_max else c_max# Make mechanism to measure max length and to test it...
                
                if sm[0].startswith('TF_'):
                    # Definition of references
                    try:
                        b_start = [ str(i[1]) for i in api_b.T.headingFromNode(api_b.L.u(match[0]+1, otype=api_b.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    except IndexError:
                        b_start = [ str(i[1]) for i in api_b.T.headingFromNode(api_b.L.u(match[0]+1, otype=api_b.F.otype.meta['availableStructure'].split(',')[-2])[0])[1:] ] + ['1']
                    try:
                        b_stop = [ str(i[1]) for i in api_b.T.headingFromNode(api_b.L.u(match[1], otype=api_b.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    except IndexError:
                        b_stop = [ str(i[1]) for i in api_b.T.headingFromNode(api_b.L.u(match[1], otype=api_b.F.otype.meta['availableStructure'].split(',')[-2])[0])[1:] ] + ['1']
                    try:
                        c_start = [ str(i[1]) for i in api_c.T.headingFromNode(api_c.L.u(match[2]+1, otype=api_c.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    except IndexError:
                        c_start = [ str(i[1]) for i in api_c.T.headingFromNode(api_c.L.u(match[2]+1, otype=api_c.F.otype.meta['availableStructure'].split(',')[-2])[0])[1:] ] + ['1']
                    try:
                        c_stop = [ str(i[1]) for i in api_c.T.headingFromNode(api_c.L.u(match[3], otype=api_c.F.otype.meta['availableStructure'].split(',')[-1])[0])[1:] ]
                    except IndexError:
                        c_stop = [ str(i[1]) for i in api_c.T.headingFromNode(api_c.L.u(match[3], otype=api_c.F.otype.meta['availableStructure'].split(',')[-2])[0])[1:] ] + ['1']
                    
                    # The prefixes (base and comp) are necessary, because Collatex uses the reference for the witness names. 
                    # If a reference is identical, Collatex runs into problems, because it cannot distinguish them anymore...
                    fref_base = f'base: {api_b.F.otype.meta["_book"]} {".".join(b_start)}' if b_start == b_stop else                                 f'base: {api_b.F.otype.meta["_book"]} {".".join(b_start)}-{b_stop[-1]}' if b_start[:-1] == b_stop[:-1] else                                 f'base: {api_b.F.otype.meta["_book"]} {".".join(b_start)}-{".".join(b_stop)}'
                    fref_comp = f'comp: {api_c.F.otype.meta["_book"]} {".".join(c_start)}' if c_start == c_stop else                                 f'comp: {api_c.F.otype.meta["_book"]} {".".join(c_start)}-{c_stop[-1]}' if c_start[:-1] == c_stop[:-1] else                                 f'comp: {api_c.F.otype.meta["_book"]} {".".join(c_start)}-{".".join(c_stop)}'
                    
                    # Define pretokenized JSON for CollateX
                    key_val = "t"
                    tokens_base_text = [ {key_val: token + " "} for token in api_b.T.text(range(base_start+1, base_stop+1), fmt='text-orig-main', descend=True).split() ]
                    tokens_comp_text = [ {key_val: token + " "} for token in api_c.T.text(range(comp_start+1, comp_stop+1), fmt='text-orig-main', descend=True).split() ]
                    witness_base_text = { "id": fref_base, "tokens": tokens_base_text }
                    witness_comp_text = { "id": fref_comp, "tokens": tokens_comp_text }
                    JSON_input = { "witnesses": [ witness_base_text, witness_comp_text ] }

                elif self.comp_path == None:
                    key_val = "t"
                    tokens_base_text = [ {key_val: token + " "} for token in list(self.base_sources[sm[0]])[base_start:base_stop] ]
                    tokens_comp_text = [ {key_val: token + " "} for token in list(self.base_sources[sm[1]])[comp_start:comp_stop] ]
                    witness_base_text = { "id": sm[0], "tokens": tokens_base_text }
                    witness_comp_text = { "id": sm[1], "tokens": tokens_comp_text }
                    JSON_input = { "witnesses": [ witness_base_text, witness_comp_text ] }
                
                else:
                    key_val = "t"
                    tokens_base_text = [ {key_val: token + " "} for token in list(self.base_sources[sm[0]])[base_start:base_stop] ]
                    tokens_comp_text = [ {key_val: token + " "} for token in list(self.comp_sources[sm[1]])[comp_start:comp_stop] ]
                    witness_base_text = { "id": sm[0], "tokens": tokens_base_text }
                    witness_comp_text = { "id": sm[1], "tokens": tokens_comp_text }
                    JSON_input = { "witnesses": [ witness_base_text, witness_comp_text ] }
                    
                # Run the collation of Collatex and print the results
#               results = collate(JSON_input, output='html', layout="horizontal", segmentation=True)
                collate(JSON_input, output='html', layout="horizontal", segmentation=True)
        
    
    def tf_initiate(self, location):
        TF = Fabric(locations=location)
        api = TF.load('', silent=True)
        allFeatures = TF.explore(silent=True, show=True)
        loadableFeatures = allFeatures['nodes'] + allFeatures['edges']
        TF.load(loadableFeatures, add=True, silent=True)
        return api
    
    
    def tf_api_clean(self, base=False, comp=False):
        if base == True:
            for api in self.base_list:
                del api
        elif comp == True:
            del TF_comp_api
        else:
            print('Nothing to clean... Please define base=True or comp=True...')
            


# In[ ]:


# # REPO = '~/github/pthu/patristics'
# VERSION = '1.0/'
# TF_DIR = os.path.expanduser(f'{REPO}/tf/{VERSION}')

# tm = Timestamp()

# x = FlexGrams(base_path=TF_DIR + 'new_testament/Brooke Foss Westcott, Fenton John Anthony Hort/New Testament - John',
# # x = FlexGrams(base_path=TF_DIR + 'patristics/Clement Of Alexandria/Paedagogus',
# # x = FlexGrams(base_path=TF_DIR + 'patristics/Clement Of Alexandria/Protrepticus',
# # x = FlexGrams(base_path=TF_DIR + 'pt/Eusebius/Historia Ecclesiastica',
# # x = FlexGrams('test',          
#            comp_path=TF_DIR + 'patristics/Clement Of Alexandria/Paedagogus',
# #            comp_path=TF_DIR + 'patristics/Eusebius/Historia Ecclesiastica',   
#            ngram=5, skip=1, number=1, context=0, distance_base=6, distance_comp=6, self_match=False, mode=2)
# # x.allign()
# x.refResult(order='base', c_levels=(0, 2, 3))
# tm.info('This is what it takes...')

# #TODO number should be present in both!


# In[ ]:


# from tf.fabric import Fabric, Timestamp
# import os
# from pprint import pprint

# REPO = '~/github/pthu/patristics'
# VERSION = '1.0/'
# TF_DIR = os.path.expanduser(f'{REPO}/tf/{VERSION}')

# TF = Fabric(locations=TF_DIR + 'patristics/Clement Of Alexandria/Paedagogus')
# # TF = Fabric(locations=TF_DIR + 'new_testament/Brooke Foss Westcott, Fenton John Anthony Hort/New Testament - John')
# api = TF.load('', silent=False)
# allFeatures = TF.explore(silent=True, show=True)
# loadableFeatures = allFeatures['nodes'] + allFeatures['edges']
# TF.load(loadableFeatures, add=True, silent=True)

# # Struct = api.F.otype.meta['availableStructure'].split(',')
# # print(Struct)
# # api.T.structureInfo()
# # print(f'structureFeats = {api.T.structureFeats}')
# # print(f'structureTypes = {api.T.structureTypes}')
# # print(f'sectionTypes = {api.T.sectionTypes}')
# # print(f'structureTypeSet = {api.T.structureTypeSet}')

# def reference(wordNode):
#     embedTuple = api.L.u(wordNode)
#     ref = ['error']
#     for node in embedTuple:
#         feature = api.F.otype.v(node)
#         if feature not in Struct:
# #             print(feature)
#             continue
#         else:
#             print(feature)
#             heading = api.T.headingFromNode(node)
#             print(heading)
#             if len(heading) > len(ref):
#                 ref = heading
#     reference = '.'.join([str(i[1]) for i in ref[:]])
#     return reference
# # print(reference(48500))
    
# # verseHeadings = [api.T.headingFromNode(v)[1:] for v in api.F.otype.s(api.F.otype.meta['availableStructure'].split(',')[-1])]
# # print(verseHeadings)

# # nodeType = api.F.otype.meta['availableStructure'].split(',')[-2]
# # nodeType = 'subsection'
# # sectionNodes = api.F.otype.s(nodeType)
# # sectionList = [api.T.headingFromNode(node)[1:] for node in sectionNodes]
# # refList = [".".join(str(i) for i in api.T.sectionFromNode(node)[1:]) for node in sectionNodes]
# # print(refList)



# # node = 11000 # Exemplaria Gratii
# # nodeTypes = api.F.otype.meta['availableStructure'].split(',')
# # # print(nodeTypes)

# # sectionList = ".".join([ str(api.Fs(level).v( i )) for level in nodeTypes for i in api.L.u(node, otype=level) ])
# # print(sectionList)

# # #Build refList (with sectionTypes)
# # nodeType = api.TF.features['otext'].metaData['sectionTypes'].split(',')[-1]
# # sectionNodes = api.F.otype.s(nodeType)
# # print(sectionNodes)
# # refList = [".".join(str(i) for i in api.T.sectionFromNode(node)[1:]) for node in sectionNodes]
# # # print(refList)

# # #Build refList (with structureTypes)
# nodeType = api.TF.features['otext'].metaData['availableStructure'].split(',')[-1]
# sectionNodes = api.F.otype.s(nodeType)
# # print(sectionNodes)
# refList = [".".join(str(i[1]) for i in api.T.headingFromNode(node)[1:]) for node in sectionNodes]
# refList = [list(str(i[1]) for i in api.T.headingFromNode(node)[1:]) for node in sectionNodes]
# # print(refList)
# refListMod = ['.'.join([st[0]] + st[2:]) for st in refList]
# # print(refListMod)

