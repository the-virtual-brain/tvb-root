from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import os

from tvb.basic.profile import TvbProfile

if __name__ == '__main__':
    # Step 1. Generate a private and a public key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    public_key = private_key.public_key()
    # The public key will be uploaded to TVB before exporting and the private key will be used for
    # decrypting the data locally

    # Step 2. Convert public key to bytes and save it so ABCUploader can use it
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.PKCS1,
    )

    public_key_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, 'public_key.pem')
    with open(public_key_path, 'wb') as f:
        f.write(pem)

    # Step 3. Convert private key to bytes and save it so it can be used later for decryption
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    private_key_path = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, 'private_key.pem')
    with open(private_key_path, 'wb') as f:
        f.write(pem)
