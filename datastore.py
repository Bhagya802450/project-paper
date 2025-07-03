from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

# Generate AES key
key = os.urandom(16)

# Example data (not multiple of 16 bytes)
data = b"PatientID:12345, BP:120/80"

# Padding (PKCS7 for 128-bit block size)
padder = padding.PKCS7(128).padder()
padded_data = padder.update(data) + padder.finalize()

# Create AES cipher in ECB mode (for demo only)
cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
encryptor = cipher.encryptor()
ct = encryptor.update(padded_data) + encryptor.finalize()

# Decrypt
decryptor = cipher.decryptor()
padded_plaintext = decryptor.update(ct) + decryptor.finalize()

# Unpad
unpadder = padding.PKCS7(128).unpadder()
plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

print("Encrypted:", ct)
print("Decrypted:", plaintext.decode())
