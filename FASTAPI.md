## Running OS2D API Service

### Docker Compose
```bash
# Please, check all environment variables at `.env` file

# Build Docker image from Docker Compose
docker-compose build

# Run container 
docker-compose up -d
```

### Docker
```bash
# Build Docker image
## Must replace all variables by their values
### See  `.env` file
docker build -t os2d:latest \
    --build-arg USER_UID=${USER_UID} \
    --build-arg USER_GID=${USER_GID} \
    --build-arg OS2D_ROOT=${WORKDIR} \
    ./Docker/

# Run container 
docker run -d \
    --name os2d \
    -p 80:80 \
    os2d:latest \
    uvicorn app:app --port 80 --host 0.0.0.0
```