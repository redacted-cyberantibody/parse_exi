# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:05:24 2018

@author: WilliamCh
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from pdf2image import convert_from_path
import os

#application_window = tk.Tk()
page_start = simpledialog.askinteger('First extraction page','Choose a starting page for image extraction')
page_end = simpledialog.askinteger('Last extraction page', 'Choose and ending page for image extraction')
fn = filedialog.askopenfilename()
fn_split = fn.split('/')
name = fn_split[-1]
folder = '/'.join(fn_split[0:-1]) + '/'
output_folder = folder + 'pdf2image_output'
try:
    os.mkdir(output_folder)
except:
    pass
output_folder = output_folder + '/'

pages = convert_from_path(fn, first_page=page_start, last_page=page_end)

for i, page in enumerate(pages):
    print(str(i) + ' out of ' + len(pages))
    page.save(folder + name.split('.')[0]+str(i+1)+'.png')