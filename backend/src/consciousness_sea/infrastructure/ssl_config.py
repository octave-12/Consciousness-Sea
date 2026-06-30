from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def generate_self_signed_cert(cert_dir: str | None = None) -> tuple[Path, Path]:
    if cert_dir is None:
        cert_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent / "certs")
    cert_path = Path(cert_dir)
    cert_path.mkdir(parents=True, exist_ok=True)

    key_file = cert_path / "key.pem"
    cert_file = cert_path / "cert.pem"

    if key_file.exists() and cert_file.exists():
        return cert_file, key_file

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_file), "-out", str(cert_file),
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost/O=consciousness-sea",
            ], check=True, capture_output=True)
            return cert_file, key_file
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("无法生成SSL证书，请手动配置或安装cryptography包")
            return cert_file, key_file

    try:
        import ipaddress as _ipaddress
        san_names = [
            x509.DNSName("localhost"),
            x509.IPAddress(_ipaddress.IPv4Address("127.0.0.1")),
        ]
    except Exception:
        san_names = [x509.DNSName("localhost")]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "consciousness-sea"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(san_names),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_file, key_file
