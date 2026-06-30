from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def generate_self_signed_cert(cert_dir: str = "certs", days: int = 365) -> None:
    cert_path = Path(cert_dir)
    cert_path.mkdir(parents=True, exist_ok=True)

    key_file = cert_path / "key.pem"
    cert_file = cert_path / "cert.pem"

    if key_file.exists() and cert_file.exists():
        print(f"SSL证书已存在: {cert_file}, {key_file}")
        return

    cmd = [
        sys.executable, "-m", "openssl_not_available",
    ]

    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", str(key_file), "-out", str(cert_file),
                "-days", str(days), "-nodes",
                "-subj", "/CN=localhost/O=ConsciousnessSea",
            ],
            check=True,
        )
        print(f"SSL证书已生成: {cert_file}, {key_file}")
    except FileNotFoundError:
        print("openssl未安装，使用Python自签名方案...")
        _generate_cert_python(cert_file, key_file, days)


def _generate_cert_python(cert_file: Path, key_file: Path, days: int) -> None:
    import base64
    import hashlib
    import os
    import struct
    import time

    key_der = _generate_rsa_key_der()
    key_file.write_bytes(_pem_encode(key_der, "RSA PRIVATE KEY"))

    cert_der = _generate_self_signed_cert_der(key_der, days)
    cert_file.write_bytes(_pem_encode(cert_der, "CERTIFICATE"))

    print(f"SSL证书已生成(Python): {cert_file}, {key_file}")


def _pem_encode(der_data: bytes, label: str) -> bytes:
    b64 = base64.b64encode(der_data).decode("ascii")
    lines = [b64[i:i+64] for i in range(0, len(b64), 64)]
    return f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n".encode()


def _generate_rsa_key_der() -> bytes:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _generate_self_signed_cert_der(key_der: bytes, days: int) -> bytes:
    from cryptography.hazmat.primitives.serialization import load_der_private_key
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    import datetime

    private_key = load_der_private_key(key_der, password=None)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ConsciousnessSea"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(private_key, hashes.SHA256())
    )
    return cert.public_bytes(x509.Encoding.DER)


def get_uvicorn_ssl_kwargs(cert_dir: str = "certs") -> dict:
    cert_file = Path(cert_dir) / "cert.pem"
    key_file = Path(cert_dir) / "key.pem"

    if cert_file.exists() and key_file.exists():
        return {
            "ssl_certfile": str(cert_file),
            "ssl_keyfile": str(key_file),
        }
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成SSL自签名证书")
    parser.add_argument("--cert-dir", default="certs", help="证书输出目录")
    parser.add_argument("--days", type=int, default=365, help="证书有效期(天)")
    args = parser.parse_args()
    generate_self_signed_cert(args.cert_dir, args.days)