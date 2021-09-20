import pyAesCrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from tvb.basic.profile import TvbProfile

if __name__ == '__main__':

    # 1. Load private key
    path_to_private_key = '/Users/Robert.Vincze/Documents/private_key.pem'
    with open(path_to_private_key, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend())

    # 2. Load encrypted password
    path_to_encrypted_password = '/Users/Robert.Vincze/Documents/encrypted_password.pem'
    with open(path_to_encrypted_password, 'rb') as f:
        encrypted_password = f.read()

    # 3. Decrypt the password using the private key
    decrypted_password = private_key.decrypt(
        encrypted_password,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    decrypted_password = decrypted_password.decode()

    # 4. Decrypt file
    file_path = '/Users/robert.vincze/Documents/TimeSeriesRegion_6ea0d6a811cf447999714a743d582939_encrypted.h5'
    decrypted_path = file_path.replace('encrypted', 'decrypted')

    pyAesCrypt.decryptFile(file_path, decrypted_path, decrypted_password, TvbProfile.current.hpc.CRYPT_BUFFER_SIZE)
