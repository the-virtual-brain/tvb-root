import os
import sys
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt_file(path_to_file, path_to_public_key, path_to_encrypted_key):
    start_extension = path_to_file.rfind('.')
    path_to_encrypt = path_to_file[:start_extension]
    extension = path_to_file[start_extension:]

    # Read the public key
    with open(path_to_public_key, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )

    # Generate the symmetric AES key and mode
    symmetric_key = os.urandom(32)
    iv = b"a" * 16

    # Generate a Cipher which uses AES algorithm
    cipher = Cipher(algorithms.AES(symmetric_key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Read the file to be encrypted as binary
    with open(path_to_file, 'rb') as f:
        content = f.read()

    # Encrypt content with symmetric key
    encrypted_content = encryptor.update(content) + encryptor.finalize()

    # Save encrypted content
    with open(path_to_encrypt + '_encrypted' + extension, 'wb') as f:
        f.write(encrypted_content)

    # Encrypt the symmetric key with the asymmetric public key
    encrypted_symmetric_key = public_key.encrypt(
        symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Save the encrypted symmetric key
    with open(os.path.join(path_to_encrypted_key, 'encrypted_symmetric_key.pem'), 'wb') as f:
        f.write(encrypted_symmetric_key)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise Exception('Three command line arguments are needed: path to the file to encrypt, path to the public key'
                        ' and path to save the encrypted data!')
    encrypt_file(sys.argv[1], sys.argv[2], sys.argv[3])
