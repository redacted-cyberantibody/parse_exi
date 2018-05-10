# lbs2kgs.py
from tkinter import *
from functools import partial
from tkinter import filedialog
from tkinter import simpledialog

from shapely.geometry import Polygon,Point,box,LinearRing,LineString
import numpy as np
import pandas as pd

import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

import parse_exi as pe

class InputFrame(Frame):
    def __init__ (self, master):
        Frame.__init__ (self, master)
        
        Label(self,text = "Exi file").grid()
        self.exi_entry = Entry(self)
        self.exi_entry.grid(row = 0, column = 1)
        self.exi_btn = Button(self,text = "Choose")
        self.exi_btn.grid(row = 0, column = 2)
        
        Label(self,text = "Source input").grid()
        self.source_entry = Entry(self)
        self.source_entry.grid(row = 1, column = 1)
        self.source_btn = Button(self,text = "Choose")
        self.source_btn.grid(row = 1, column = 2)
        
        Label(self,text = "Room input").grid()
        self.room_entry = Entry(self)
        self.room_entry.grid(row = 2, column = 1)
        self.room_btn = Button(self,text = "Choose")
        self.room_btn.grid(row = 2, column = 2)
        
        self.loadbtn = Button(self,text = "Load input files")
        self.loadbtn.grid(columnspan = 3)
        
        self.dosebtn = Button(self,text = "Dose to rooms")
        self.dosebtn.grid(columnspan = 3)
        
        self.leadbtn = Button(self,text = "Lead requirement")
        self.leadbtn.grid(columnspan = 3)
        
        self.roombtn = Button(self,text = "Show room")
        self.roombtn.grid(row = 7, column = 0)        
        
        self.workloadbtn = Button(self,text = "Workload")
        self.workloadbtn.grid(row = 7, column = 1) 
        
        self.ogpbtn = Button(self,text = "OGP stats")
        self.ogpbtn.grid(row = 7, column = 2)
        
        self.reportbtn = Button(self,text = "Save plots")
        self.reportbtn.grid(columnspan = 3)  
        
        self.xraybarrbtn = Button(self,text = "Create XRAYBARR spectrums from Exi")
        self.xraybarrbtn.grid(columnspan = 3)
        
        self.verbosebtn = Button(self,text = "Produce full report")
        self.verbosebtn.grid(columnspan = 3)
        
class TextFrame(Frame):
    def __init__ (self, master):
        Frame.__init__ (self, master)
        Label(self,text = "Room name").grid()
        self.room_name = Entry(self)
        self.room_name.grid(row = 0, column = 1)
    
        Label(self,text = "Conservatism factor").grid()
        self.conservatism = Entry(self)
        self.conservatism.grid(row = 1, column = 1)
        
        Label(self,text = "EXI rescale factor (override)").grid()
        self.exi_rescale = Entry(self)
        self.exi_rescale.grid(row = 2, column = 1)
         
class DataFrame(Frame):
    def __init__ (self, master):
        Frame.__init__ (self, master)
        self.textlabel = Label(self)
        self.textlabel.pack()
        self.datalabel = Label(self)
        self.datalabel.pack()
        
class View(Frame):
    def __init__(self,master = None):
        Frame.__init__ (self, master)
        self.pack()
        self.text_frame = TextFrame(master)
        self.text_frame.pack(side=LEFT)
        self.input_frame = InputFrame(master)
        self.input_frame.pack(side=LEFT)
        self.data_frame = DataFrame(master)
        self.data_frame.pack(side=LEFT)

class Controller:
    def __init__(self):
        
        self.root = Tk()
        self.view = View(self.root)
        
        self.exi_fn = StringVar()
        self.conservatism = StringVar()
        self.exi_rescale = StringVar()
        self.room_name = StringVar()
        self.source_fn = StringVar()
        self.room_fn = StringVar()
        
        self.exi_fn.set('sample_exi_log.csv')
        self.source_fn.set('input_sources.csv')
        self.room_fn.set('input_rooms.csv')
        
        self.view.input_frame.exi_entry.configure(textvariable = self.exi_fn)
        self.view.input_frame.source_entry.configure(textvariable = self.source_fn)
        self.view.input_frame.room_entry.configure(textvariable = self.room_fn)
        self.view.text_frame.room_name.configure(textvariable = self.room_name)
        self.view.text_frame.conservatism.configure(textvariable = self.conservatism)
        self.view.text_frame.exi_rescale.configure(textvariable = self.exi_rescale)
        
        self.view.input_frame.exi_btn.bind("<Button-1>",lambda x: self.choose_file(self.exi_fn))
        self.view.input_frame.room_btn.bind("<Button-1>",lambda x: self.choose_file(self.room_fn))
        self.view.input_frame.source_btn.bind("<Button-1>",lambda x: self.choose_file(self.source_fn))
        
        self.view.input_frame.loadbtn.bind("<Button-1>",self.load_input_files)
        self.view.input_frame.dosebtn.bind("<Button-1>",self.get_dose_to_rooms)
        self.view.input_frame.leadbtn.bind("<Button-1>",self.get_lead_required)
        self.view.input_frame.xraybarrbtn.bind("<Button-1>",self.export_for_xraybarr)
        self.view.input_frame.verbosebtn.bind("<Button-1>",self.verbose_report)
        self.view.input_frame.reportbtn.bind("<Button-1>",self.produce_report)
        self.view.input_frame.roombtn.bind("<Button-1>",self.show_room)
        self.view.input_frame.workloadbtn.bind("<Button-1>",self.show_workload)
        self.view.input_frame.ogpbtn.bind("<Button-1>",self.show_OGP_stats)
        
        self.output_text = StringVar()
        self.view.data_frame.textlabel.configure(textvariable = self.output_text)
        self.output_data = StringVar()
        self.view.data_frame.datalabel.configure(textvariable = self.output_data)
        
        self.output_text = StringVar()
        self.view.data_frame.textlabel.configure(textvariable = self.output_text)
        self.output_data = StringVar()
        self.view.data_frame.datalabel.configure(textvariable = self.output_data)
        
        
    def choose_file(self, stringvar):
        fn = filedialog.askopenfilename(title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
        stringvar.set(fn)
        print(fn)
        
    def choose_folder(self):
        fn = filedialog.askdirectory (title = "Select location for created spectrum files")
        return fn
    
    def update_output(self,heading='',data=''):
        self.output_text.set(heading)
        self.output_data.set(data)
        
    def load_input_files(self,event):
        try:
            self.D = pe.Dose(exi_fn = self.exi_fn.get(),
                             dfr_fn = self.room_fn.get(),
                             dfs_fn = self.source_fn.get())
            self.update_output('Loaded input files')
        except Exception as e:
            self.update_output('Failed to load input files',e)
        
    def get_dose_to_rooms(self,event):
        try:
            self.D
        except Exception as e:
            self.update_output('input files not loaded')
        a = self.D.calculate_dose()
        self.update_output('Weekly dose (uGy)',
                            a.sum())
    
    def get_lead_required(self,event):
        try:
            self.D
        except Exception as e:
            self.update_output('input files not loaded')
        a,b,c,d = self.D.get_lead_req(iterations = 3)
        
        #self.D.dfr[['added_attenuation','wall_weight']]
        self.update_output('Lead required (mm)',a)
        
    def export_for_xraybarr(self,event):
        output_folder = self.choose_folder()+'/'
        if not output_folder:
            self.update_output('No output folder selected, try again')
            return
        try:
            pe.make_xraybarr_spectrum_set(self.exi_fn.get(), room_name=self.room_name.get(), output_folder=output_folder)
            self.update_output('Exported for XRAYBARR','folder: '+output_folder)
        except Exception as e:
            self.update_output('Could not export to XRAYBARR:',str(e))
        try:
            self.D.export_distancemap(output_folder)
        except:
            pass
    
    def verbose_report(self,event):
        output_folder = self.choose_folder()+'/'
        if not output_folder:
            self.update_output('No output folder selected, try again')
            return
        try:
            self.D
        except Exception as e:
            self.update_output('input files not loaded', str(e))
            return
        R = pe.Report(self.D, output_folder, self.room_name.get())
        R.full_report()
    
    def produce_report(self,event):
        output_folder = self.choose_folder()+'/'
        if not output_folder:
            self.update_output('No output folder selected, try again')
            return
        #try:
        #R = pe.Report(self.D, output_folder=output_folder, room_name=self.room_name.get())
        #R.full_report()
        #except:
        #    self.update_output('Could not produce reports')
    
    def show_room(self,event):
        pe.Report(self.D).show_room()
    
    def show_workload(self,event):
        pe.Report(self.D).source_workload_plots()
    
    def show_OGP_stats(self,event):
        pe.Report(self.D).OGP_workload_plot()
    
    
    def run(self):
        self.root.title('Shielding dose calculator')
        self.root.deiconify()
        self.root.mainloop()


        
        
        
if __name__ == '__main__':
    c = Controller()
    c.run()