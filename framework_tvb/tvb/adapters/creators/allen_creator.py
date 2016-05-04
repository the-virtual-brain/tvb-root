# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
The adapters in this module create new connectivities from the Allen Institute
data using their SDK.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import os.path
import numpy as np
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.core.entities.storage import dao
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.volumes import Volume
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


class AllenConnectomeBuilder(ABCAsynchronous):
    "Handler for uploading a mouse connectivity from Allen dataset using AllenSDK."

    _ui_name = "Allen connectivity uploader"
    _ui_description = "Import mouse connectivity from Allen database (tracer experiments)"

    #TRANSGENIC_OPTIONS = [
    #    {'name': 'No', 'value': 'False'},
    #    {'name': 'Yes', 'value': 'True'}
    #]

    RESOLUTION_OPTIONS = [
        {'name': '10', 'value': '10'},
        {'name': '25', 'value': '25'},
        {'name': '50', 'value': '50'},
        {'name': '100', 'value': '100'}
    ]

    WEIGHTS_OPTIONS = [
        {'name': '(projection density)/(injection density)', 'value': '1'},
        {'name': 'projection density', 'value': '2'},
        {'name': 'projection energy', 'value': '3'},
    ]

    def get_input_tree(self):
        return [# {'name': 'TransgenicLine', 'type': 'select', 'label': 'Transgenic line :',
                # 'required': True, 'options': self.TRANSGENIC_OPTIONS, 'default': 'False'},

                {'name': 'resolution', 'type': 'select',
                 'label': 'Spatial resolution (micron)',
                 'description': 'Resolution of the data that you want to download to construct the volume and the connectivity (micron) :',
                 'required': True, 'options': self.RESOLUTION_OPTIONS, 'default': '100'},

                {'name': 'weighting', 'type': 'select', 'label': 'Definition of the weights of the connectivity :',
                 'required': True, 'options': self.WEIGHTS_OPTIONS, 'default': '1'},

                {'name': 'inf_vox_thresh', 'type': 'float',
                 'label': 'Min infected voxels',
                 'description': 'To build the volume and the connectivity select only the areas where there is at least one infection experiment that had infected more than ... voxels in that areas.',
                 'required': True, 'default': '50'},

                {'name': 'vol_thresh', 'type': 'float',
                 'label': 'Min volume',
                 'description': 'To build the volume and the connectivity select only the areas that have a volume greater than (micron^3): ',
                 'required': True, 'default': '1000000000'}]

    def get_output(self):
        return [Connectivity, RegionVolumeMapping, Volume]


    def launch(self, resolution, weighting, inf_vox_thresh, vol_thresh):
        resolution = int(resolution)
        weighting = int(weighting)
        inf_vox_thresh = float(inf_vox_thresh)
        vol_thresh = float(vol_thresh)

        project = dao.get_project_by_id(self.current_project_id)
        manifest_file = self.file_handler.get_allen_mouse_cache_folder(project.name)
        manifest_file = os.path.join(manifest_file, 'mouse_connectivity_manifest.json')
        cache = MouseConnectivityCache(resolution=resolution, manifest_file=manifest_file)

        Vol, annot_info = cache.get_annotation_volume()
        ontology = cache.get_ontology()

        ist2e = DictionaireBuilder(cache, False)

        projmaps = DownloadAndConstructMatrix(cache, weighting, ist2e, False)

        # This method cleans the file projmaps in 4 steps
        projmaps = pmsCleaner(projmaps)

        #Taking only brain regions whose volume is greater than vol_thresh
        projmaps = AreasVolumeTreshold(projmaps, vol_thresh, resolution, Vol, ontology)

        # Taking only brain regions where at least one exp has infected more than N voxel (where N is inf_vox_thresh)
        projmaps = AreasVoxelTreshold(cache, projmaps, inf_vox_thresh, Vol, ontology)

        [Order, KeyOrd] = CreateFileOrder(projmaps, ontology)

        SC = ConstructingSC(projmaps, Order, KeyOrd)

        [centres, names] = Construct_centres(ontology, Order, KeyOrd, Vol)

        tract_lengths = ConstructTractLengths(centres)
        # For the Vol I should calculate before Parents and Granparents:
        [UniqueParents, UniqueGranParents] = ParentsAndGranParentsVolumeCalculator(Order, KeyOrd, Vol, ontology)

        # Vol indexed between 0:(N-1)areas, -1=background and areas not in the selected parcellation
        Vol_parcel = MouseBrainVisualizer(Vol, Order, KeyOrd, UniqueParents, UniqueGranParents, ontology, projmaps)

        # results: Connectivity & RegionVolumeMapping
        #Connectivity
        result_connectivity = Connectivity(storage_path=self.storage_path)
        result_connectivity.centres = centres
        result_connectivity.region_labels = names
        result_connectivity.weights = SC
        result_connectivity.tract_lengths = tract_lengths

        # Volume
        result_volume = Volume(storage_path=self.storage_path)
        result_volume.origin = [[0.0, 0.0, 0.0]]
        result_volume.voxel_size = [resolution, resolution, resolution]
        #result_volume.voxel_unit= micron
        # Region Volume Mapping
        result_rvm = RegionVolumeMapping(storage_path=self.storage_path)
        result_rvm.volume = result_volume
        result_rvm.array_data = Vol_parcel
        result_rvm.connectivity = result_connectivity
        result_rvm.title = "Volume mouse brain "
        result_rvm.dimensions_labels = ["X", "Y", "Z"]
        return [result_connectivity, result_rvm, result_volume]

    def get_required_memory_size(self, **kwargs):
        return -1

    def get_required_disk_size(self, **kwargs):
        return -1



# This method construct the dictionary of the experiments that you want to download
def DictionaireBuilder(tvb_mcc,TransgenicLine):
    # open up a list of all of the experiments
    all_experiments = tvb_mcc.get_experiments(dataframe=True, cre=TransgenicLine)
    #build dict of injection structure id to experiment list
    ist2e = {}
    for eid in all_experiments.index:
        for ist in all_experiments.ix[eid]['injection-structures']:
             isti = ist['id']
             if isti not in ist2e:
                ist2e[isti]=[]
             ist2e[isti].append(eid)
    return ist2e


def DownloadAndConstructMatrix(tvb_mcc,weighting,ist2e,TransgenicLine):
    projmaps = {}
    if weighting==3: # case in which I download projection energy
        for isti, elist in ist2e.items():
            projmaps[isti] = tvb_mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=ist2e.keys(), #summary_structure_ids,
                parameter='projection_energy')
            print 'injection site id', isti, ' has ', len(elist), ' experiments with pm shape ', projmaps[isti]['matrix'].shape
    else: # in the other 2 cases for sure I should download projection density:
        for isti, elist in ist2e.items():
            projmaps[isti] = tvb_mcc.get_projection_matrix(
                     experiment_ids=elist,
                     projection_structure_ids=ist2e.keys(), #summary_structure_ids,
                     parameter='projection_density')
            print 'injection site id', isti, ' has ', len(elist), ' experiments with pm shape ', projmaps[isti]['matrix'].shape
        if weighting==1: # in this case I should also download injection density
            injdensity = {}
            all_experiments = tvb_mcc.get_experiments(dataframe=True, cre=TransgenicLine)
            for exp_id in all_experiments['id']:
                injD=tvb_mcc.get_injection_density(exp_id, file_name=None)
                #I check that all experiments have only an injection sites (only 3 coordinates)
                #thus we can sum on the injection matrix since contains only the injection density for a single injection
                injdensity[exp_id]=(np.sum(injD[0])/np.count_nonzero(injD[0]))
                print 'Experiment id', exp_id, ', the total injection density is ', injdensity[exp_id]
            #now in this case I want PD/ID and like this I will modify the file projmaps:
            for inj_id in range(len(projmaps.values())):
                index=0
                for exp_id in projmaps.values()[inj_id]['rows']:
                    projmaps.values()[inj_id]['matrix'][index]=projmaps.values()[inj_id]['matrix'][index]/injdensity[exp_id]
                    index+=1
    return projmaps

#This method cleans the file projmaps in 4 steps
def pmsCleaner(projmaps):
    def get_structure_id_set(pm):
        return set([c['structure_id'] for c in pm['columns']])
    sis0=get_structure_id_set(projmaps[515])
    # 1) All the target sites are the same for all the injection sites? If not I will remove the corresponding injection sites
    DifferentTargetSet=[]
    for inj in projmaps.keys():
        sis_i=get_structure_id_set(projmaps[inj])
        if len(sis0.difference(sis_i))!=0:
            projmaps.pop(inj, None)
            DifferentTargetSet.append(inj)
    if len(DifferentTargetSet)!=0:
        print 'There are',len(DifferentTargetSet),' injection sites that do not project in the same set of target sites of the others injection sites'
        print'Their ID is', DifferentTargetSet,'. I removed that injection sites.'

    else:
        print 'All the matrix have the same kind of target sites, great!'

    # 2) All the injection sites are also target sites?
    #    If not we will remove that injection sites
    NotTarg=[]
    for inj in projmaps.keys():
        if inj not in sis0:
            del projmaps[inj]
            NotTarg.append(inj)

    if len(NotTarg)!=0:
        print 'There are', len(NotTarg),'inj sites that were not targ sites; their ID is', NotTarg
        print 'I removed that injection sites; now all the inj sites are also target sites'
    else:
        print 'All the injection sites are also target sites.'

    # 3) All the target sites are also injection sites?
    #    if not remove that targets (from the columns and from the matrix)
    if len(sis0)==len(projmaps.keys()):
        print 'Great! All the target sites are also inj sites'
    else:
        print 'There are', len(sis0)-len(projmaps.keys()),'area that are target but not injection, I will removed them from the target sites list and in the matrices.'
        for indexinj in range(len(projmaps.values())):
            indextarg=-1
            while len(projmaps.values()[indexinj]['columns'])!=(3*len(projmaps.keys())): #there is -3 since for each id-target I have 3 regions since I have 3 hemisphere to consider
                indextarg+=1
                if projmaps.values()[indexinj]['columns'][indextarg]['structure_id'] not in projmaps.keys():
                    del projmaps.values()[indexinj]['columns'][indextarg]
                    projmaps.values()[indexinj]['matrix']=np.delete(projmaps.values()[indexinj]['matrix'],indextarg,1)
                    indextarg=-1

    # 4) Exclude the areas (in inj and target) that have NaN values (in all the experiments; only one exp with no-Nan values, and I keep the region)
    # this is a choice that I made by myself.
    Nan_id={}
    for inj in projmaps.keys():
        M=projmaps[inj]['matrix']
        for targ in range(M.shape[1]):
            if np.isnan(M[0,targ]):
                if all([np.isnan(M[exp,targ]) for exp in range(M.shape[0])]):
                    Nan_id[inj]=[]
                    Nan_id[inj]=projmaps[inj]['columns'][targ]['structure_id']
    #Remove Nan areas from the injection list
    for remove in Nan_id.values():
        projmaps.pop(remove, None)
    #Remove Nan areas from targe list (columns+matrix)
    for indexinj in range(len(projmaps.keys())):
        indextarg=-1
        PreviousSize= len(projmaps.values()[indexinj]['columns'])
        while len(projmaps.values()[indexinj]['columns'])!=(PreviousSize-3*len(Nan_id)): #3 hemispheres
            indextarg+=1
            COL=projmaps.values()[indexinj]['columns'][indextarg]
            if COL['structure_id'] in Nan_id.values():
               del projmaps.values()[indexinj]['columns'][indextarg]
               projmaps.values()[indexinj]['matrix']=np.delete(projmaps.values()[indexinj]['matrix'],indextarg,1)
               indextarg=-1

    # Summary of what we have obtain
    lenleft=0
    lenright=0
    rl=0
    for targ in projmaps.values()[0]['columns']:
        if targ['hemisphere_id']==1:
            lenleft+=1
        if targ['hemisphere_id']==2:
            lenright+=1
        else:
            rl+=1
    print 'Now we have clean everything and we end up with:'
    print len(projmaps.keys()), 'injection sites (in the right hemisphere)'
    print lenright,' target sites in the ipsilateral hemisphere'
    print lenleft,' target sites in the controlateral hemisphere.'

    return projmaps



#Taking only brain regions where at least one exp has infected more than inf_vox_thresh Voxel
# what we can download is the Injection fraction: fraction of pixels belonging to manually annotated injection site (http://help.brain-map.org/display/mouseconnectivity/API)
#when this fraction is greater than inf_vox_thresh I consider that area in
def AreasVoxelTreshold (tvb_mcc,projmaps,inf_vox_thresh, Vol,ontology):
    IDok=[]
    cc=0
    for ID in projmaps.keys():
        for exp in projmaps[ID]['rows']:
            inj_f, inf_info=tvb_mcc.get_injection_fraction(exp)
            inj_voxels=np.where( inj_f >= 0) #coordinates of the injected voxels with inj fraction>=1
            id_infected=Vol[inj_voxels] #id of area located in Vol where injf>=0 (thus the id of the infected structure)
            myid_infected=np.where(id_infected==ID)
            inj_f_id=0     #fraction of injected voxels for the ID I am currently examinating       
            for index in myid_infected[0]:
                inj_f_id+=inj_f[inj_voxels[0][index], inj_voxels[1][index], inj_voxels[2][index]]
                if inj_f_id>=inf_vox_thresh:
                    break
            if inj_f_id>=inf_vox_thresh:
                IDok.append(ID)
                break
            else:
                child=list(ontology.get_child_ids( ontology[ID].id ))
                while len(child)!=0:
                    if child[0] in Vol:
                        myid_infected=np.where(id_infected==child[0])
                        if len(myid_infected[0])!=0:
                            for index in myid_infected[0]:
                                inj_f_id+=inj_f[inj_voxels[0][index], inj_voxels[1][index], inj_voxels[2][index]]
                                if inj_f_id>=inf_vox_thresh:
                                    break
                            if inj_f_id>=inf_vox_thresh:
                                IDok.append(ID)
                                break
                    else:
                        if len(ontology.get_child_ids(ontology[child[0]].id))!=0:
                            child.extend(ontology.get_child_ids( ontology[child[0]].id ))
                    child.remove(child[0])
            if inj_f_id>=inf_vox_thresh:
                break
        cc+=1
        print 'I have', len(IDok), 'and I have examinating area', cc,'over',len(projmaps.keys())            
    #Remove areas that are not in IDok from the injection list
    for checkid in projmaps.keys():
        if checkid not in IDok:
            projmaps.pop(checkid,None)
    #Remove areas that are not in IDok from target list (columns+matrix)
    for indexinj in range(len(projmaps.values())):
        indextarg=-1
        while len(projmaps.values()[indexinj]['columns'])!=(len(IDok)*3): #I have 3 hemispheres
            indextarg+=1
            if projmaps.values()[indexinj]['columns'][indextarg]['structure_id'] not in IDok:
                del projmaps.values()[indexinj]['columns'][indextarg]
                projmaps.values()[indexinj]['matrix']=np.delete(projmaps.values()[indexinj]['matrix'],indextarg,1)
                indextarg=-1
    return projmaps




#Taking only brain regions that have a volume greater than vol_thresh
def AreasVolumeTreshold (projmaps,vol_thresh,resolution,Vol,ontology):
    treshold=(vol_thresh)/(resolution**3)
    IDok=[]
    cc=0
    for ID in projmaps.keys():
        n_voxels=np.where(Vol==ID)
        tot_voxels=(n_voxels[0].shape)[0]
        if tot_voxels>treshold:
            IDok.append(ID)
        else: #I will check for child
            child=list(ontology.get_child_ids( ontology[ID].id ))
            while len(child)!=0:
                if child[0] in Vol:
                    n_voxels=np.where(Vol==child[0])
                    tot_voxels+=(n_voxels[0].shape)[0]
                    if tot_voxels>treshold:
                        IDok.append(ID)
                        break
                else:
                    if len(ontology.get_child_ids(ontology[child[0]].id))!=0:
                        child.extend(ontology.get_child_ids( ontology[child[0]].id ))
                child.remove(child[0])
        cc+=1
        print 'I have', len(IDok), 'and I have examinating area', cc,'over',len(projmaps.keys())
     #Remove areas that are not in IDok from the injection list
    for checkid in projmaps.keys():
        if checkid not in IDok:
            projmaps.pop(checkid,None)
    #Remove areas that are not in IDok from target list (columns+matrix)
    for indexinj in range(len(projmaps.values())):
        indextarg=-1
        while len(projmaps.values()[indexinj]['columns'])!=(len(IDok)*3): #I have 3 hemispheres
            indextarg+=1
            if projmaps.values()[indexinj]['columns'][indextarg]['structure_id'] not in IDok:
                del projmaps.values()[indexinj]['columns'][indextarg]
                projmaps.values()[indexinj]['matrix']=np.delete(projmaps.values()[indexinj]['matrix'],indextarg,1)
                indextarg=-1
    return projmaps

# Now that I have clean up totally file projmaps I start to constrcut the matrix/centres/tractlengths.
# this function create file order and keyord that will be our method to link the SC order and the ID in the Allen Atlas
def CreateFileOrder (projmaps,ontology):
    Order={}
    for index in range(len(projmaps)):
        TargetKey=projmaps.values()[0]['columns'][index]['structure_id']
        ont=ontology[TargetKey]
        Order[ont.loc[TargetKey]['graph_order']]=[TargetKey]
        Order[ont.loc[TargetKey]['graph_order']].append(ont.loc[TargetKey]['name'])

    KeyOrd=Order.keys()
    KeyOrd.sort()
    return Order, KeyOrd


def ConstructingSC(projmaps,Order,KeyOrd):
    lenright=len(projmaps.keys())
    SC=np.zeros((lenright,2*lenright), dtype=float)
    row=-1
    for graph_ord_inj in KeyOrd:
        row+=1
        ID_inj= Order[graph_ord_inj][0]
        TARG=projmaps[ID_inj]['columns']
        M=projmaps[ID_inj]['matrix']
        #average on the experiments (NB: if there are NaN values not average!)
        if np.isnan(np.sum(M)):
            M_temp=np.zeros((M.shape[1],1), dtype=float)
            for i in range(M.shape[1]):
                if np.isnan(sum(M[:,i])):
                    occ=0
                    for jj in range(M.shape[0]):
                        if M[jj,i]==M[jj,i]: #since nan!=nan
                            occ+=1
                            M_temp[i,0]=M_temp[i,0]+M[jj,i]
                    M_temp[i,0]=M_temp[i,0]/occ
                else:
                    M_temp[i,0]=sum(M[:,i])/M.shape[0]
            M=M_temp
        else:
            M=(np.array([sum(M[:,i]) for i in range(M.shape[1])])/(M.shape[0]))
        #now we will order the target
        col=-1
        for graph_ord_targ in KeyOrd:
            col+=1
            ID_targ=Order[graph_ord_targ][0]
            for index in range(len(TARG)):
                if TARG[index]['structure_id']==ID_targ:
                    if TARG[index]['hemisphere_id']==2:
                        SC[row,col]=M[index]
                    if TARG[index]['hemisphere_id']==1:
                        SC[row,col+lenright]=M[index]


    # I will save the complete matrix (both left and right inj):
    FirstQuarter=SC[:,:(SC.shape[1]/2)]
    SecondQuarter=SC[:,(SC.shape[1]/2):]
    SC_down=np.concatenate((SecondQuarter,FirstQuarter), axis=1)
    SC=np.concatenate((SC,SC_down),axis=0)
    SC=SC/(np.amax(SC)) #normalize the matrix
    return SC

def Construct_centres(ontology, Order,KeyOrd,Vol):
    centres=np.zeros((len(KeyOrd)*2,3), dtype=float)
    names=[]
    row=-1
    vol_r=Vol[:,:,:Vol.shape[2]/2]
    for graph_ord_inj in KeyOrd:
        ID=Order[graph_ord_inj][0]
        Coord=[0,0,0]
        print 'I am calculating centre of area:', Order[graph_ord_inj]
        if ID in vol_r: #Check if the area is in the annotation volume
            xyz=np.where(vol_r==ID)
            Coord[0]=np.mean(xyz[0])
            Coord[1]=np.mean(xyz[1])
            Coord[2]=np.mean(xyz[2])
        else: #The area is not in the annotation volume, check for child
            print 'That area is not in the annotation volume, I will check for child'
            child=list(ontology.get_child_ids( ontology[ID].id ))
            FinestDivision=[]
            while len(child)!=0:
                if child[0] in vol_r:
                    FinestDivision.append(child[0])
                else:
                    if len(ontology.get_child_ids(ontology[child[0]].id))!=0:
                        child.extend(ontology.get_child_ids( ontology[child[0]].id ))
                child.remove(child[0])
            for IDchild in FinestDivision:
                if IDchild in vol_r:
                    xyz=np.where(vol_r==IDchild)
                    Coord[0]=np.mean(xyz[0])
                    Coord[1]=np.mean(xyz[1])
                    Coord[2]=np.mean(xyz[2])
        row+=1
        centres[row,:]=Coord
        Coord[2]=(Vol.shape[2])-Coord[2]
        centres[row+len(KeyOrd),:]=Coord
        n=Order[graph_ord_inj][1]
        Right='Right '
        Right+=n
        Right=str(Right)
        names.append(Right)
    for graph_ord_inj in KeyOrd:
        n=Order[graph_ord_inj][1]
        Left='Left '
        Left+=n
        Left=str(Left)
        names.append(Left)
    return centres, names


def ConstructTractLengths(centres):
    lenhalf=len(centres)/2
    tracts=np.zeros((lenhalf,len(centres)), dtype=float)
    for inj in range(lenhalf):
        Inj=centres[inj]
        for targ in range(lenhalf):
            Targ_r=centres[targ]
            Targ_l=centres[targ+lenhalf]
            tracts[inj,targ]=np.sqrt((Inj[0]-Targ_r[0])**2+(Inj[1]-Targ_r[1])**2+(Inj[2]-Targ_r[2])**2)
            tracts[inj,targ+lenhalf]=np.sqrt((Inj[0]-Targ_l[0])**2+(Inj[1]-Targ_l[1])**2+(Inj[2]-Targ_l[2])**2)
    # I will save the complete matrix (both left and right inj):
    FirstQuarter=tracts[:,:(tracts.shape[1]/2)]
    SecondQuarter=tracts[:,(tracts.shape[1]/2):]
    tracts_down=np.concatenate((SecondQuarter,FirstQuarter), axis=1)
    tracts=np.concatenate((tracts,tracts_down),axis=0)
    return tracts


#Now the Vol:
def ParentsAndGranParentsVolumeCalculator(Order,KeyOrd,Vol,ontology):
    Parents=[] #Here I will put the parents of the areas that I have in my parcel
    GranParents=[]
    VV=[] #Here I will put the volume of the areas that I have in myparcel
    IndVec=[] #Here I will put the index of the eigenvector of the areas that I have in my parcel
    indexvec=0
    for graph_ord_inj in KeyOrd:
        ID=Order[graph_ord_inj][0]
        ont=ontology[ID]
        Parents.append(ont.loc[ID]['parent_structure_id'])
        gg=ont.loc[ID]['parent_structure_id']
        gg=int(gg)
        ont=ontology[gg]
        GranParents.append(ont.loc[gg]['parent_structure_id'])
        IndVec.append(indexvec)
        indexvec+=1
        if ID in Vol: #To calculate the vol of each area I should look also at the child
            NVoxels=np.where(Vol==ID)
            TotVoxels=NVoxels[0].shape
            VV.append(TotVoxels)
        else:
            TotVoxels=0
            print 'That area is not in the annotation volume, I will check for child'
            child=list(ontology.get_child_ids( ontology[ID].id ))
            while len(child)!=0:
                print child
                if child[0] in Vol:
                    NVoxels=np.where(Vol==child[0])
                    bb=NVoxels[0].shape
                    TotVoxels+=bb[0]
                else:
                    if len(ontology.get_child_ids(ontology[child[0]].id))!=0:
                        child.extend(ontology.get_child_ids( ontology[child[0]].id ))
                child.remove(child[0])
            VV.append(TotVoxels)
    #I will order Parents, granparents, Index vec according to the volume
    Parents=[parents for (vv,parents) in sorted(zip(VV,Parents))]
    GranParents=[granparents for (vv,granparents) in sorted(zip(VV,GranParents))]
    IndVec=[iid for (vv,iid) in sorted(zip(VV,IndVec))]
    k=len(Parents)
    UniqueParents={}#Unique parents will be a dictionary with keys the parent id and as value the index vec of the region in my parcellation that has that parent id
    for P in reversed(Parents):
        k-=1
        if P not in UniqueParents.keys():
            UniqueParents[P]=IndVec[k]
    k=len(GranParents)
    UniqueGranParents={}#Unique parents will be a dictionary with keys the parent id and as value the index vec of the region in my parcellation that has that parent id
    for P in reversed(GranParents):
        k-=1
        if np.isnan(P)==0:
            if P not in UniqueGranParents.keys():
                UniqueGranParents[P]=IndVec[k]
    return UniqueParents, UniqueGranParents




def MouseBrainVisualizer(Vol,Order,KeyOrd,UniqueParents,UniqueGranParents, ontology, projmaps):
    Nareas=len(KeyOrd)*2
    Vec = np.arange(Nareas).reshape(Nareas,)
    Vec=Vec+1 #indicizzato fra 1 e Nareas
    Vec=Vec*(10**(-(1+int(np.log10(Nareas))))) #now, all the Vec components are always less than one
    vol_r=Vol[:,:,:(Vol.shape[2]/2)]
    vol_r=vol_r.astype(np.float64)
    vol_l=Vol[:,:,(Vol.shape[2]/2):]
    vol_l=vol_l.astype(np.float64)
    index_vec=0 #this is the index of the vector
    left=len(Vec)/2
    for graph_ord_inj in KeyOrd:
        ID=Order[graph_ord_inj][0]
       # print 'I am associating the value of Vec to area:', Order[graph_ord_inj]
        if ID in vol_r: #Check if the area is in the annotation volume
            vol_r[vol_r == ID] = Vec[index_vec]
            vol_l[vol_l == ID] =Vec[index_vec+left]
        child=list(ontology.get_child_ids( ontology[ID].id ))
        while len(child)!=0:
            if (child[0] in vol_r) and (child[0] not in projmaps.keys()):
                vol_r[vol_r == child[0]] = Vec[index_vec]
                vol_l[vol_l == child[0]] = Vec[index_vec+left]
            if len(ontology.get_child_ids(ontology[child[0]].id))!=0:
                child.extend(ontology.get_child_ids( ontology[child[0]].id ))
            child.remove(child[0])
        index_vec+=1 # index of vector
    vol_parcel=np.concatenate((vol_r, vol_l), axis=2)
    #Now we have a problem: since the parcellation is reduced I do not have a lot of areas plotted, so I can plot the areas that are not plotted
    #using the following trick:
    #I will plot with the same value of the eig the areas that are brother (it means that they have the same parent in the ontology)
    #but since a lot of the areas that I have in the parcellation have the same parent and I want an univoca relation
    #for each parent I will associate only the biggest (bigger volume) child in my parcellation
    bool_idx = (vol_parcel > np.amax(Vec))  # Find the elements of vol_parcel that are yet not associated to a value of the eigenvector
    # We use boolean array indexing to construct a rank 1 array consisting of the elements of a corresponding to the True values of bool_idx
    not_assigned=np.unique(vol_parcel[bool_idx])
    vol_r=vol_parcel[:,:,:(Vol.shape[2]/2)]
    vol_r=vol_r.astype(np.float64)
    vol_l=vol_parcel[:,:,(Vol.shape[2]/2):]
    vol_l=vol_l.astype(np.float64)
    #In this loop I will look at the parents in the next at the granparents
    count=1
    for ID in not_assigned:
        #print 'Esamino area',count,'di aree',len(not_assigned)
        count+=1
        ID=int(ID)
        ont=ontology[ID]
        PP=ont.loc[ID]['parent_structure_id']
        if np.isnan(PP):
            k=0 #If the area has not parent it will return a NaN value thus I will not enter in the while loop
        else:
            k=1
        while k==1:
            PP=int(PP)
            if PP in UniqueParents.keys():
                vol_r[vol_r == ID] = Vec[UniqueParents[PP]]
                vol_l[vol_l == ID] = Vec[UniqueParents[PP]+left]
                k=0
            else:
                ont=ontology[PP]
                PP=ont.loc[PP]['parent_structure_id']
                if np.isnan(PP):
                    k=0
    vol_parcel=np.concatenate((vol_r, vol_l), axis=2)
    #Now I will look at the granparents for the areas not yet assigned
    bool_idx = (vol_parcel > np.amax(Vec))  # Find the elements of vol_parcel that are yet not associated to a value of the eigenvector
    # We use boolean array indexing to construct a rank 1 array consisting of the elements of a corresponding to the True values of bool_idx
    not_assigned=np.unique(vol_parcel[bool_idx])
    vol_r=vol_parcel[:,:,:(Vol.shape[2]/2)]
    vol_r=vol_r.astype(np.float64)
    vol_l=vol_parcel[:,:,(Vol.shape[2]/2):]
    vol_l=vol_l.astype(np.float64)
    count=1
    for ID in not_assigned:
        #print 'Esamino area',count,'di aree',len(not_assigned)
        count+=1
        ID=int(ID)
        ont=ontology[ID]
        PP=ont.loc[ID]['parent_structure_id']
        if np.isnan(PP):
            k=0 #If the area has not parent it will return a NaN value thus I will not enter in the while loop
        else:
            k=1
        while k==1:
            PP=int(PP)
            if PP in UniqueGranParents.keys():
                vol_r[vol_r == ID] = Vec[UniqueGranParents[PP]]
                vol_l[vol_l == ID] = Vec[UniqueGranParents[PP]+left]
                k=0
            else:
                ont=ontology[PP]
                PP=ont.loc[PP]['parent_structure_id']
                if np.isnan(PP):
                    k=0
    vol_parcel=np.concatenate((vol_r, vol_l), axis=2)
    vol_parcel[vol_parcel>=1]=0 #set all the areas not in the parcellation to 0 (as the background that is zero)
    vol_parcel=vol_parcel*(10**(1+int(np.log10(Nareas)))) #ritorno alla indicizzazione fra uno e Nareas
    vol_parcel=vol_parcel-1 #with this operation background and areas not in parcellation will be -1 and all the others with the index as in python (betweeen 0 and Nareas-1)
    vol_parcel = np.round(vol_parcel)
    return vol_parcel