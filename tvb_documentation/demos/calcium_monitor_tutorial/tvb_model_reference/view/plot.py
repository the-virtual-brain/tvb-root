#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 09:14:44 2021

edited by nuuria
"""

from matplotlib.tri import Triangulation
from pylab import *
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def multiview(cortex,hemisphere_left,hemisphere_right,data, fig, suptitle='', figsize=(15, 10), CB_position=[0.27, 0.8, 0.5, 0.05], CB_orientation='horizontal', CB_fontsize=10, CB_label='FR (Hz)', fixed_legend=[], **kwds):
    cs = cortex
    vtx = cs.vertices
    tri = cs.triangles
    rm = cs.region_mapping
    x, y, z = vtx.T
    lh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_left ]))]
    lh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_left])]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_right ]))]
    rh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_right])]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T

    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[argsort(tz)]),
    }


    def plotview(i, j, k, viewkey, z=None, zmin=None, zmax=None, zthresh=None, suptitle='', shaded=True, cmap=plt.cm.coolwarm, viewlabel=False):
        v = views[viewkey]
        ax = subplot(i, j, k)
        if z is None:
            z = rand(v.x.shape[0])
        if not viewlabel:
            axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.1}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        tc.set_clim(vmin=zmin, vmax=zmax)
        

        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=40)
        if viewlabel:
            xlabel(viewkey)
        return tc

    if len(fixed_legend):
        zmin=fixed_legend[0]
        zmax=fixed_legend[1]
        plotview(2, 3, 1, 'lh-lateral', data, zmin, zmax, **kwds)
        plotview(2, 3, 4, 'lh-medial', data, zmin, zmax,  **kwds)
        plotview(2, 3, 3, 'rh-lateral', data, zmin, zmax,  **kwds)
        plotview(2, 3, 6, 'rh-medial', data, zmin, zmax, **kwds)
        tc = plotview(1, 3, 2, 'both-superior', data, zmin, zmax,  suptitle=suptitle, **kwds)
    
    else:
        plotview(2, 3, 1, 'lh-lateral', data, **kwds)
        plotview(2, 3, 4, 'lh-medial', data, **kwds)
        plotview(2, 3, 3, 'rh-lateral', data, **kwds)
        plotview(2, 3, 6, 'rh-medial', data, **kwds)
        tc = plotview(1, 3, 2, 'both-superior', data, suptitle=suptitle, **kwds)
    subplots_adjust(left=0.0, right=CB_position[0], bottom=0.0, top=1.0, wspace=0, hspace=0)
    cax = fig.add_axes(CB_position)
    fig.colorbar(tc,cax=cax,orientation=CB_orientation,label=CB_label)
    cax.tick_params(axis='y',labelsize=CB_fontsize)
    plt.title = title
    return cax


def animation(cortex,hemisphere_left,hemisphere_right,data,begin,end,file_name='./test.gif'):

    my_dpi=100 # resolution
    # parameter fig
    text_fontsize = 20.0
    title_fontsize = 20.0
    fig = plt.figure(figsize=(900/my_dpi, 900/my_dpi), dpi=my_dpi) # size of windows

    def update_fig(i):
        print(i,end=' ' )
        fig.clf() # clear figure
        multiview(cortex,hemisphere_left,hemisphere_right,data[i],fig=fig,shaded=False)
        return [fig]
    anim = manimation.FuncAnimation(fig, update_fig,frames=np.arange(begin,end,1), blit=True)
    anim.save(file_name)
    plt.close('all')
    
    
def animation_nuu(cortex,hemisphere_left,hemisphere_right,data,file_name='./movie.mp4',fps=35,my_dpi=100,factor=10, label='FR (Hz)', use_old=False):  
    
    
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        import re
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
 
    def remove_dir(dirname):
        import os
        files = os.listdir(dirname)
        for f in files:
            os.remove(dirname+'/'+f)
        os.rmdir(dirname)
        
    if not use_old:
        try:
            remove_dir('movie')
        except:
            print('nothing to remove')
    
    import os
    import moviepy.video.io.ImageSequenceClip
    image_folder='movie'
    if use_old:
        images = os.listdir(image_folder)
    else:
        os.mkdir(image_folder)
    
    
    
    mi,ma = np.min(data),np.max(data)
    r_mima = abs(0.1*(ma-mi))
    mi -=r_mima
    ma -=2*r_mima
    
    for ii in range(np.shape(data)[0]):
        print(ii,end=' ' )
        if use_old:
            done = ('movie%d.png' % ii) in images
        else:
            done = False
        if not done:
            fig = plt.figure( dpi=my_dpi)
            cax = multiview(cortex, hemisphere_left, hemisphere_right, data[ii], fig=fig, suptitle='t = %1.3f'% round(ii*0.1*factor*1e-3,3)+'s', CB_position=[0.8, 0.1, 0.02, 0.8], CB_orientation='vertical', CB_label = label, shaded=False, fixed_legend=[mi,ma])

            fig.savefig("movie/movie%d.png" % ii)
            plt.close()
            
    image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
    image_files.sort(key=alphanum_key)
    
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid errors associated with truncated image files

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(file_name) 

        
    
def prepare_surface_regions(parameters, h5_filename='Connectivity_nuria_v1.h5', zip_filename= 'Cortex.zip', vertices_to_region_filename = 'vertices_to_region.npy', region_map_filename = 'region_map.txt'):
    import os
    from tvb.simulator.lab import cortex as ct
    from tvb.simulator.lab import region_mapping as rm
    from tvb.simulator.lab import surfaces as surf
    from tvb.simulator.lab import connectivity


    # path of the zip file 
    path = parameters.parameter_connection_between_region['path']
    if not os.path.isdir(path):
        mm = [i for i in range(len(path)) if '/' in path[i]][-1]
        path = path[:mm+1]
        

    # connectivity of TVB
    conn = connectivity.Connectivity().from_file(path + h5_filename)
    conn.configure()
    
    # surface of TVB
    surface_cortex = surf.Surface().from_file(path + zip_filename)
    hemispheres_left = np.where(conn.hemispheres )[1]
    hemispheres_right = np.where( np.logical_not( conn.hemispheres ) )[1]
    
    # remake the region mapping 
    v_to_reg = np.load(path + vertices_to_region_filename)
    region_mapping_data = []
    for point in surface_cortex.vertices:
        index = np.where(np.logical_and(np.logical_and(np.abs(v_to_reg[:,0] - point[0])<1e-3, np.abs(v_to_reg[:,1] - point[1])<1e-3), np.abs(v_to_reg[:,2] - point[2])<1e-3) )[0]
        if len(index) != 1:
            raise Exception('bad index')
        region_mapping_data.append(int(v_to_reg[index,3]))

    # create the region mapping and the cortex
    region_mapping = rm.RegionMapping(array_data=np.concatenate([region_mapping_data],axis=0), connectivity=conn, surface=surface_cortex)
    surface_cortex.vertices[:,0] *=-1  # mirror of x axis (frontal on top)
    
    cortex = ct.Cortex().from_file(source_file=os.path.join(path,zip_filename), region_mapping_file=os.path.join(path,region_map_filename))
    cortex.region_mapping_data = region_mapping 

    return region_mapping, cortex, conn, hemispheres_left, hemispheres_right
    
    

def multiview_one(cortex,hemisphere_left,hemisphere_right,region,data, fig, suptitle='', title='', figsize=(15, 10), **kwds):
    cs = cortex
    vtx = cs.vertices
    tri = cs.triangles
    rm = cs.region_mapping
    x, y, z = vtx.T
    
    lh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_left ]))]
    lh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_left])]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[np.unique(np.concatenate([ np.where(rm[tri] == i)[0] for i in hemisphere_right ]))]
    rh_vtx = vtx[np.concatenate([np.where(rm == i )[0] for i in hemisphere_right])]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T
    
    data = np.zeros_like(data)
    if type(region)==list:
        for r in region:
            data[rm == r] = 10.0
    else:
        data[rm == region ] =  10.0

    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[argsort(tz)]),
    }
    

    def plotview(i, j, k, viewkey, z=None, zlim=None, zthresh=None, suptitle='', shaded=True, cmap=plt.cm.coolwarm, viewlabel=False):
        v = views[viewkey]
        ax = subplot(i, j, k)
        if z is None:
            z = rand(v.x.shape[0])
        if not viewlabel:
            axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.1}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        if zlim:
            tc.set_clim(vmin=-zlim, vmax=zlim)
        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=24)            
        if viewlabel:
            xlabel(viewkey)
        return tc

    plotview(2, 3, 1, 'lh-lateral', data, **kwds)
    plotview(2, 3, 4, 'lh-medial', data, **kwds)
    plotview(2, 3, 3, 'rh-lateral', data, **kwds)
    plotview(2, 3, 6, 'rh-medial', data, **kwds)
    plotview(1, 3, 2, 'both-superior', data, suptitle=suptitle, **kwds)
    
    subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)
    
    if title:
        plt.gcf().suptitle(title)
    


if __name__ == '__main__':
    animation()