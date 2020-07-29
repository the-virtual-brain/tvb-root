import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt_file(path_to_file, path_to_public_key, path_to_encrypted_key):
    start_extension = path_to_file.rfind('.')
    path_to_encrypt = path_to_file[:start_extension]
    extension = path_to_file[start_extension:]

    # Reading the public key
    with open(path_to_public_key, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )

    # Generating symmetrical AES key
    symmetric_key = os.urandom(32)
    iv = b"a" * 16

    # Generating Cipher
    cipher = Cipher(algorithms.AES(symmetric_key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # reading the file as binary
    f = open(path_to_file, 'rb')
    content = f.read()
    f.close()

    # Encrypt with symmetric key
    encrypted_content = encryptor.update(content) + encryptor.finalize()

    f = open(path_to_encrypt + '_encrypted' + extension, 'wb')
    f.write(encrypted_content)
    f.close()

    # Encrypt the symmetric key with the asymmetric public key
    encrypted_symmetric_key = public_key.encrypt(
        symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    f = open(os.path.join(path_to_encrypted_key, 'encrypted_symmetric_key.pem'), 'wb')
    f.write(encrypted_symmetric_key)
    f.close()


if __name__ == '__main__':
    encrypt_file() # TODO: Make this script use command line arguments, instead of specifying parameters here
