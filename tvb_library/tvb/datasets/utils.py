import requests
from pathlib import Path
import hashlib
import urllib
from tqdm import tqdm

"""
functions related to hashes functions

"""

USER_AGENT = "TVB_ROOT/TVB_LIBRARY"

def calculate_md5(file_path:Path, chunk_size:int =1024) -> str :
    """
    A function to calculate the md5 hash of a file.

    """
    m = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda : f.read(chunk_size), b""):
            m.update(chunk)
    return m.hexdigest();




def calculate_sha256(file_path:Path, chunk_size:int =1024) -> str:
    """ 
 /  A function to calculate the sha256 hash of a file
    """
    s = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda : f.read(chunk_size), b""):
            s.update(chunk)
    return s.hexdigest();


def calculate_sha1(file_path:Path, chunk_size:int=1024)->str:
    s = hashlib.sha1()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda : f.read(chunk_size), b""):
            m.update(chunk)

    return s.hexdigest()



def calculate_sha224(file_path:Path, chunk_size:int=1024)->str:
    s = hashlib.sha224()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda : f.read(chunk_size), b""):
            m.update(chunk)

    return s.hexdigest()


def calculate_sha384(file_path:Path, chunk_size:int=1024)->str:
    s = hashlib.sha384()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda : f.read(chunk_size), b""):
            m.update(chunk)

    return s.hexdigest()

#
def calculate_sha512(file_path:Path, chunk_size:int=1024):
    s = hashlib.sha512()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda : f.read(chunk_size), b""):
            m.update(chunk)

    return s.hexdigest()
#.

# okay there are some stuff which would consider generic SHA hash; link -https://github.com/zenodo/zenodo/issues/1985#issuecomment-796882811



AVAILABLE_HASH_FUNCTIONS = {"md5": calculate_md5, "sha1": calculate_sha1,"sha224":calculate_sha224, "sha256":calculate_sha256, "sha384":calculate_sha384, "sha512": calculate_sha512} # can extend this further 


def convert_to_pathlib(file_path: str) ->Path:
    """ 
    convert the file_path to Path datatype
    """

    if (type(file_path)!= Path):
        return Path(file_path)
    return file_path



#should we keep a way to download a file without having to check the checksum? 

def check_integrity(file_loc, checksum:str, hash_function="md5")->bool:
    """ 
    This function checks if the file at `file_loc` has same checksum.
    """

    if hash_function not in AVAILABLE_HASH_FUNCTIONS.keys():
        raise AttributeError(f"incorrect hash function value, must be one of the md5, sha1,sha224,sha256, sha384, sha512, received  {hash_functio}")
    
    if hash_function== "md5":
        return calculate_md5(file_loc)==checksum

    if hash_function == "sha1":
        return calculate_sha1(file_loc) == checksum
   
    if hash_function == "sha224":
        return calculate_sha224(file_loc) == checksum
   
    if hash_function == "sha256":
        return calculate_sha256(file_loc) == checksum

    if hash_function == "sha384":
        return calculate_sha384(file_loc) == checksum
   
    if hash_function == "sha512":
        return calculate_sha512(file_loc) == checksum




def download_file(url, checksum, hash_function, root):
    if hash_function not in AVAILABLE_HASH_FUNCTIONS.keys():
        raise AttributeError(f"incorrect hash function value, must be one of the md5, sha1,sha224,sha256, sha384, sha512, received  {hash_functio}")
 
    root = Path(root)

    if (not root.is_dir()):
        root.mkdir(parents=True)

    file_name = url.split("/")[-1]
    file_loc = root/file_name

    if (file_loc.is_file() and check_integrity(file_loc, checksum, hash_function)):
        print(f"File {file_name} already downloaded at location {file_loc}")
        return 

    _urlretrieve(url, file_loc)

    #ToDO : what to do when the hash of the downloaded file doesnt match with the online value? discard the file ? warning the user? both? 

    print(f"file {file_loc} downloaded successfully")



# following functions are inspired from the torchvision.
def _save_response_content(
    content,
    destination,
    length= None,
) :
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(url, file_loc, chunk_size = 1024 * 32):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        _save_response_content(iter(lambda: response.read(chunk_size), b""), file_loc, length=response.length)
