#!/bin/bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout quantum.key -out quantum.crt -subj "/CN=__REDACTED__" \
  -addext "subjectAltName=DNS:__REDACTED__,DNS:*.__REDACTED__,IP:192.168.100.32"