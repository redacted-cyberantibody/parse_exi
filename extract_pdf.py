# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:05:24 2018

@author: WilliamCh
"""

from tkinter import filedialog
from pdf2image import convert_from_path

fn = filedialog.askopenfilename()
fn_split = fn.split('/')
name = fn_split[-1]
folder = '/'.join(fn_split[0:-1]) + '/'

pages = convert_from_path(fn)

for i, page in enumerate(pages):
    page.save(folder + name.split('.')[0]+str(i+1)+'.tif')