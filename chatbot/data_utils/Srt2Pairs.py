#!/usr/bin/env python

import json
import numpy
import re
import sys
import os
import argparse


FLAGS = None


#no punctuation for the source sentence
nopun_exp = r"[0-9]+|[']*[\w]+"
#complete expression for response sentence
expression = r"[0-9]+|[']*[\w]+|[.]+|[,?!\"()]"


max_length = 20

zero_token = "[non]"
unknown_token = "[unk]"
start_token = "[beg]"
end_token = "[end]"

class Srt2Pairs:

        def __init__(self, input_file, dict_filename, text_filename):
            self.input_file = input_file
            self.dict_filename = dict_filename
            self.text_filename = text_filename
            
            if os.path.exists(dict_filename): os.remove(dict_filename)
            if os.path.exists(text_filename): os.remove(text_filename)               
              
            self.build_dict()
            self.write_dict()
            self.write_pairs()
        
        def build_dict(self):

            from Queue import PriorityQueue
            token_count_dict={}

            with open(self.input_file) as f:
                    
                print "Creating Dictionary..."
                line_count=0
                for line in f:
                    token_list = re.findall(expression, line.lower())
                    for token in token_list:
                        if token not in token_count_dict:
                            token_count_dict[token]=1
                        else:
                            token_count_dict[token]+=1
                    line_count+=1

                print "Lines in the Dataset: " + str(line_count)
            
            q=PriorityQueue()
            for t in token_count_dict:
                q.put([-token_count_dict[t], t])
            self.token_dict = {}
            #add special token
            self.token_dict[zero_token]=0
            self.token_dict[unknown_token]=1
            self.token_dict[start_token]=2
            self.token_dict[end_token]=3
            token_index = 4
            
            token_count_dict={}
            
            #priority queue
            while (not q.empty()):
                get=q.get_nowait()
                self.token_dict[get[1]]=token_index
                token_index+=1
        
        
        def write_pairs(self):
            with open(self.input_file) as f:  
                line_count=0
                last_exist=False
                last_list=[]


            	text_file = open(self.text_filename, "a")

                for line in f:
                       
                   line_count+=1
   
                   #parse input
                   token_list = re.findall(expression, line.lower())
                   token_no_pun = re.findall(nopun_exp, line.lower())
   
                   #discard the sentences that are too long.
                   if len(token_list) > max_length or len(token_list)==0:
                       last_list=[]
                       last_exist=False
                       continue
   
                   new_list=[]
                   new_list_nopun = []
                           
                   for token in token_list:
                       if token in self.token_dict:
                               new_list.append(self.token_dict[token])
                       else:
                           new_list.append(1)

                   #create sentences with no punctuation for source
                   for token in token_no_pun:
                       if token in self.token_dict:
                           new_list_nopun.append(self.token_dict[token])
                       else:
                           new_list_nopun.append(1)
   
                   #ignore [unk] tokens and continue
                   if len(token_no_pun)==0:
                       last_exist = False
                       continue
   
                   # Write the pairs to file
                   if(last_exist==True):
                       text_file.write(json.dumps([last_list,new_list])+"\n")
   
                   # source previously created sentences that have no punctuations
                   last_list=new_list_nopun
                   last_exist=True
       
                print "Pairs created" 
                text_file.close()
        
        
        def write_dict(self):

        	dict_file = open(self.dict_filename, "a")

        	dict_file.write(json.dumps(self.token_dict))
        	
        	dict_file.close()


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple python class to convert a subtitle .srt file into data pairs and dict rapresentation')
    parser.add_argument('--dataset', type=str, required=True, default='', help='Path to the srt file')
    FLAGS = parser.parse_args()
    s = Srt2Pairs(FLAGS.dataset, "dizionario.json", "enc_text.txt")
