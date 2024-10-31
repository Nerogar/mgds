from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
import csv
import re
import os
import functools


class DropTags(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_in_name: str,
            enabled_in_name: str,
            probability_in_name: float,     #from 0-1, probability dropout will be applied to either each tag in list or entire section of caption after kept token
            dropout_mode_in_name: str,      #whether to dropout all tags after kept token (FULL), randomly drop any of the tags after kept with equal probability (RANDOM), or randomly drop any tag after kept with lower probability near the start (RANDOM WEIGHTED)
            special_tags_in_name: str,      #list of delimiter-separated tags or file path to txt/csv file which will be used as white/blacklist
            special_tag_mode_in_name: str,  #whether the special tags are used as a WHITELIST (any tags in the list will never be dropped) or a BLACKLIST (only tags in the list can de dropped, all others kept)
            delimiter_in_name: str,
            regex_enabled_in_name: str,     #whether to process special tags with regex
            keep_tags_count_in_name: str,
            text_out_name: str,
    ):
        super(DropTags, self).__init__()
        self.text_in_name = text_in_name
        self.enabled_in_name = enabled_in_name
        self.probability_in_name = probability_in_name
        self.dropout_mode_in_name = dropout_mode_in_name
        self.special_tags_in_name = special_tags_in_name
        self.special_tag_mode_in_name = special_tag_mode_in_name
        self.delimiter_in_name = delimiter_in_name
        self.regex_enabled_in_name = regex_enabled_in_name
        self.keep_tags_count_in_name = keep_tags_count_in_name
        self.text_out_name = text_out_name

    def length(self) -> int:
        return self._get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name, self.enabled_in_name, self.probability_in_name, self.dropout_mode_in_name, self.special_tags_in_name, self.special_tag_mode_in_name, self.delimiter_in_name, self.keep_tags_count_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]
    
    #convert special_tags to list depending on whether it's a newline-separated csv/txt file or a delimiter-separated string
    #cached to reduce file read operations
    @functools.lru_cache
    def get_special_tags(self, sptags, delim):
        if (sptags.endswith(".txt") and os.path.isfile(sptags)):
            with open(sptags) as special_tags_file:
                return [line.rstrip('\n') for line in special_tags_file]
        elif (sptags.endswith(".csv") and os.path.isfile(sptags)):
            with open(sptags, 'r') as special_tags_file:
                splist = []
                for row in csv.reader(special_tags_file):
                    splist.append(row[0])
                return splist
        else:
            return [tag.strip() for tag in sptags.split(delim)]

    #parse regex expressions, create new special list based on matches   
    def parse_regex(self, splist_in, taglist):
        splist_out = []
        regex_spchars = set(".^$*+?!\{\}\[\]|()\\")
        for c in splist_in:
            if any((a in regex_spchars) for a in c):    #only do regex matching if tag contains special character
                c = c.replace("\)", "\\\\\)")
                c = c.replace("\(", "\\\\\(")   #hopefully fix issues caused by "\(\)" syntax without affecting other regex
                r = re.compile(c)
                for s in taglist:
                    if r.fullmatch(s):
                        splist_out.append(s)
            else:
                splist_out.append(c)
        return splist_out    
    
    #change probability evaluated against random() depending on mode
    def probability_weighted(self, p, mode, i, len):
        if mode == "RANDOM":
            return float(p)
        elif mode == "RANDOM WEIGHTED":
            return float(p*(i/len))
        else:   #catch errors
            return float(p)
    
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.text_in_name, index)
        delimiter = self._get_previous_item(variation, self.delimiter_in_name, index)
        keep_tags_count = self._get_previous_item(variation, self.keep_tags_count_in_name, index)
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        probability = self._get_previous_item(variation, self.probability_in_name, index)
        dropout_mode = self._get_previous_item(variation, self.dropout_mode_in_name, index)
        special_tag_mode = self._get_previous_item(variation, self.special_tag_mode_in_name, index)
        special_tags = self._get_previous_item(variation, self.special_tags_in_name, index)
        regex_enabled = self._get_previous_item(variation, self.regex_enabled_in_name, index)
        rand = self._get_rand(variation, index)

        if enabled and (probability > 0):
            #convert inputs to lists and set up final output list (pruned) and white/blacklist (special)
            tags = [tag.strip() for tag in text.split(delimiter)]
            keep_tags = tags[:keep_tags_count]
            dropout_tags = tags[keep_tags_count:]
            pruned_tags = []
            special_tags_prelist = []
            special_tags_list = []
            
            #load special tags from file or string
            special_tags_prelist = self.get_special_tags(special_tags, delimiter)

            #match any regex expressions in the special tags prelist to tags in the dropout list and create a new list of all matching tags
            #if any sort of (tag weighting:1.2) or {other|special|syntax} is added in the future this may need to be changed
            if regex_enabled:
                special_tags_list = self.parse_regex(special_tags_prelist, dropout_tags)
            else:
                special_tags_list = special_tags_prelist

            #remove duplicates
            special_tags_list = list(set(special_tags_list))
                    

            if (dropout_mode == "FULL") and (rand.random() < probability):
                #keep only whitelist tags if random < probability, or any non-blacklist tags
                for s in dropout_tags:
                    if (special_tag_mode == "WHITELIST" and s in special_tags_list):
                        pruned_tags.append(s)
                    elif (special_tag_mode == "BLACKLIST" and not(s in special_tags_list)):
                        pruned_tags.append(s)
                    elif (special_tag_mode == "NONE"):
                        pruned_tags.append(s)
            elif (dropout_mode.startswith("RANDOM")):     
                #iterate through dropout_tags and add to pruned_tags if random > probability
                if (special_tag_mode == "WHITELIST"):
                    for i, s in enumerate(dropout_tags):
                        if (rand.random() > self.probability_weighted(probability, dropout_mode, i, len(dropout_tags)) or (s in special_tags_list)):
                            pruned_tags.append(s)
                elif (special_tag_mode == "BLACKLIST"):
                    for i, s in enumerate(dropout_tags):
                        if (rand.random() > self.probability_weighted(probability, dropout_mode, i, len(dropout_tags)) or not(s in special_tags_list)):
                            pruned_tags.append(s)
                elif (special_tag_mode == "NONE"):
                    for i, s in enumerate(dropout_tags):
                        if (rand.random() > self.probability_weighted(probability, dropout_mode, i, len(dropout_tags))):
                            pruned_tags.append(s)
            else:
                #avoid dropping any captions if dropout_mode isn't an expected value, or if in "FULL" mode and random > probability
                pruned_tags = dropout_tags

            tags = keep_tags + pruned_tags
            text = delimiter.join(tags)

        return {
            self.text_out_name: text
        }
