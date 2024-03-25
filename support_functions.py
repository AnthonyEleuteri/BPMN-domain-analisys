#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Notebook con funzioni di supporto globali
import numpy as np
import pandas as pd
import ast
import re
import chardet

# Funzione che dato un 'path' restituisce il suo encoding.  
def get_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        content=f.read()
        result = chardet.detect(content)
        return result['encoding']

