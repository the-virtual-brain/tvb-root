from .zenodo import Zenodo, Record


class TVB_Data:

    conceptid = ""
    
    def __init__(self, version= "2.7", ):

            recid = Zenodo().get_version_info(self.conceptid)[version]
            self.rec = Zenodo.get_record(recid)


    def download(self):
        
        self.rec.download()

    def fetch_data(self):
        pass




