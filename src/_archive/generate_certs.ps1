# Generate SSL certificates for quantum bridge
$cert = New-SelfSignedCertificate -Subject "CN=QuantumBridge" -DnsName "QuantumBridge" -KeyAlgorithm RSA -KeyLength 2048 -NotAfter (Get-Date).AddYears(2) -CertStoreLocation "Cert:\LocalMachine\My"

# Export certificate and private key
$pwd = ConvertTo-SecureString -String "quantum432" -Force -AsPlainText
$certPath = "d:\WindSurf\quantum-core\quantum_cert.pfx"
Export-PfxCertificate -Cert $cert -FilePath $certPath -Password $pwd

# Convert to PEM format for Python
$pemPath = "d:\WindSurf\quantum-core\quantum_cert.pem"
$keyPath = "d:\WindSurf\quantum-core\quantum_key.pem"

# Use OpenSSL to convert
openssl pkcs12 -in $certPath -clcerts -nokeys -out $pemPath -password pass:quantum432
openssl pkcs12 -in $certPath -nocerts -nodes -out $keyPath -password pass:quantum432
