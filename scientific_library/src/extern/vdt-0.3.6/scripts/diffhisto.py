# !/usr/bin/env python

import optparse
import re

#-------------------------------------------------------------------------------

def getTypeLength(cmpfile):
  # read header, determine type length in bits
  typelength = 0
  typelength_s =""
  cmpfile.readline()
  tmp = cmpfile.readline()
  if tmp[0] == "D":
    typelength = 64
    typelength_s = "double"
  else:
    typelength = 32 
    typelength_s = "float"
  return typelength,typelength_s

#-------------------------------------------------------------------------------  
  
def getYaxisRange(name):
  lr = -5000
  hr = 5000
  if "Asin" in name or "Acos" in name:
    lr = -1
    hr = 1
  if "Log" in name or "Isqrt" in name:
    lr = 0   
  if "Exp" in name:
    lr = -705
    hr = 705
  if "Expf" in name:
    lr = -85
    hr = 85
  return (lr,hr)

#-------------------------------------------------------------------------------      
def getFilenamesFromDir(nick,dirname):
  import os
  return filter (lambda filename: re.match("comparison__%s__(.*).txt"%nick,filename), os.listdir(dirname))
  
#-------------------------------------------------------------------------------    

def fill_histos(cmpfile,histo1D,histo2D):
  #read lines and process them   
  while 1:
    line = cmpfile.readline()
    if line == "":
      break
    m=re.match("(.*) (.*) (.*) (.*) (.*)",line)
    inputval=float(m.group(5))
    db=int(m.group(4))
    # fill histograms
    histo1D.Fill(db)
    histo2D.Fill(inputval,db)  

#-------------------------------------------------------------------------------    

def compare(nick,dirname):
  import ROOT
  
  # Some globals for the style
  ROOT.gROOT.SetStyle("Plain")
  ROOT.gROOT.SetBatch()
  ROOT.gStyle.SetPalette(1)
   # Do it for all function variants
  ofile = ROOT.TFile("%s_histos.root" %nick,"RECREATE")
  ofile.cd()
  
  filenames = getFilenamesFromDir(nick,dirname)
  print "%s files found." %len(filenames)
  for filename in filenames:      
    print "Studying " + filename
    cmpfile = open(filename)
    typelength,typelength_s =  getTypeLength(cmpfile)
    
    m=re.match("comparison__%s__(.*).txt"%nick,filename)
    fcn_name=m.group(1)
    

    # read rest of header   
    for i in range(1,5):
        cmpfile.readline()
    
    # xaxis range
    xmin,xmax=(-0.5,typelength+0.5)
    xNbins=typelength+1
    
    # set up Root 1D histo
    dbhisto = ROOT.TH1F("Diffbit_"+fcn_name, 
                  fcn_name+" diffbit for "+nick+";Diffbit;#",
                  xNbins,
                  xmin,xmax)
    dbhisto.SetLineColor(ROOT.kBlue)
    dbhisto.SetLineWidth(2)
    dbhisto.GetYaxis().SetTitleOffset(1)

    # setup diffbit VS input histogram
    ymin,ymax=getYaxisRange(fcn_name)

    dbVSinhisto = ROOT.TH2F("DiffVsInput_"+fcn_name, 
                    fcn_name+" diffbit vs input for "+nick+";Input;Diffbit",
                    100,ymin,ymax,
                    xNbins,xmin,xmax)
    
    fill_histos(cmpfile,dbhisto,dbVSinhisto)
    
    # draw and save 1D histogram
    dbcanvas = ROOT.TCanvas("dbcanv_%s"%fcn_name, "%s diffbit for %s canvas" %(fcn_name,nick),600,600)
    dbcanvas.cd()
    dbcanvas.SetLogy()
    dbhisto.Draw()
    dbcanvas.Print("%s_%s_%s_dbhisto.png"%(nick,typelength_s,fcn_name))
    dbhisto.Write()


    # draw and save 2D histogram
    dbVSincanvas = ROOT.TCanvas("dbVSincanv_%s"%fcn_name, "%s diffbit for %s canvas" %(fcn_name,nick),600,600)
    dbVSincanvas.cd()
    dbVSinhisto.Draw("COLZ")
    dbVSincanvas.Print("%s_%s_%s_dbVSinhisto.png"%(nick,typelength_s,fcn_name))
    
    dbVSinhisto.Write()
    
  ofile.Close()    

#-------------------------------------------------------------------------------      
    
def create_parser():
  import sys
  # set up cmd options
  cmdParser = optparse.OptionParser(usage="%prog -n=<nick>")
  cmdParser.add_option("-n","--nick",dest="nick",help="Determines which comparison files should be used.",default="")
  cmdParser.add_option("-d","--dir",dest="dirname",help="Directory to be looked at.",default="./")
  #parse options and retrieve needed parameters
  (options,args) = cmdParser.parse_args()
  
  if(options.nick == ""):
    cmdParser.print_help()
    sys.exit(1)

  return options

#-------------------------------------------------------------------------------    
  
if __name__ == "__main__":
  options = create_parser()
  compare(options.nick,options.dirname)
  
      