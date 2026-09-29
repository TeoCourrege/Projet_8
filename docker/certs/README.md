# Certificats racine supplémentaires (optionnel)

Tout fichier `*.crt` (format PEM) placé ici est ajouté aux autorités de
confiance de l'image Docker au build (système, pip, uv).

Utile derrière un proxy d'entreprise qui inspecte le TLS (erreur
`x509: certificate signed by unknown authority`). Les `*.crt` sont ignorés
par git : ce dossier ne contient normalement que ce README dans le dépôt, et
le build fonctionne sans certificat (CI, machine personnelle).

Exemple sur macOS :

```bash
security find-certificate -c "<nom de la CA>" -p > docker/certs/corporate-root-ca.crt
```

Avec Colima, la VM Docker doit aussi faire confiance à la CA (pour `docker pull`) :

```bash
colima ssh -- sudo sh -c "cp $PWD/docker/certs/corporate-root-ca.crt /usr/local/share/ca-certificates/ && update-ca-certificates && systemctl restart docker"
```
