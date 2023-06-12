from tvb.datasets import Zenodo, Record
from pathlib import Path


class TestZenodo(BaseTestCase):

    def test_get_record(self):

        zenodo = Zenodo()
        rec = zenodo.get_record("7574266")

        assert type(rec) == Record 
        assert rec.data["doi"] == "10.5281/zenodo.7574266"

        del rec 
        del zenodo


    def test_get_versions(self):

        zenodo = Zenodo()
        versions = zenodo.get_versions_info()

        assert type(versions) == dict
        assert versions == {'2.0.1': '3497545', '1.5.9.b': '3474071', '2.0.0': '3491055', '2.0.3': '4263723', '2.0.2': '3688773', '1.5.9': '3417207', '2.7': '7574266'}

        del zenodo
        del versions

class TestRecord(BaseTestCase):


    def test_download(self):

        zen = Zenodo()

        rec = zenodo.get_record("7574266")

        rec.download()

        for file_name, file_path in rec.file_loc:
            assert Path(file_path).is_file()


