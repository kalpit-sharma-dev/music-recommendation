import os
import binascii
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt(plaintext: str, key_hex: str) -> str:
    """
    AES-256-GCM Encryption
    Matches the Golang implementation:
    - Hex encoded key
    - Random nonce
    - Output = nonce + ciphertext (hex encoded)
    """

    try:
        key = binascii.unhexlify(key_hex)
    except Exception as e:
        raise ValueError(f"hex decode: {e}")

    if len(key) != 32:
        raise ValueError("key must be 32 bytes (AES-256)")

    aesgcm = AESGCM(key)

    # GCM standard nonce size = 12 bytes
    nonce = os.urandom(12)

    ciphertext = aesgcm.encrypt(
        nonce=nonce,
        data=plaintext.encode("utf-8"),
        associated_data=None
    )

    # Combine nonce + ciphertext
    combined = nonce + ciphertext

    # Return hex encoded string
    return binascii.hexlify(combined).decode("utf-8")


def decrypt(ciphertext_hex: str, key_hex: str) -> str:
    """
    AES-256-GCM Decryption
    Matches the Golang implementation
    """

    try:
        key = binascii.unhexlify(key_hex)
    except Exception as e:
        raise ValueError(f"hex decode: {e}")

    if len(key) != 32:
        raise ValueError("key must be 32 bytes (AES-256)")

    aesgcm = AESGCM(key)

    try:
        data = binascii.unhexlify(ciphertext_hex)
    except Exception as e:
        raise ValueError(f"ciphertext hex decode: {e}")

    nonce_size = 12

    if len(data) < nonce_size:
        raise ValueError("ciphertext too short")

    nonce = data[:nonce_size]
    ciphertext = data[nonce_size:]

    try:
        plaintext = aesgcm.decrypt(
            nonce=nonce,
            data=ciphertext,
            associated_data=None
        )
    except Exception as e:
        raise ValueError(f"decrypt: {e}")

    return plaintext.decode("utf-8")


# Example Usage
if __name__ == "__main__":

    # 32-byte AES256 key in HEX
    key_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    plaintext = "Hello AES256 GCM"

    encrypted = encrypt(plaintext, key_hex)
    print("Encrypted:", encrypted)

    decrypted = decrypt(encrypted, key_hex)
    print("Decrypted:", decrypted)
