<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
 *
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 *
http://www.apache.org/licenses/LICENSE-2.0
 *
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Helm Chart Deployment

## Create Required Secrets

These secrets allow your Kubernetes applications to securely access NVIDIA resources and your database without hardcoding credentials in your application code or container images.

To deploy the secrets required by the CA-RAG Blueprint:

> **Note**
> If using `microk8s`, prepend the `kubectl` commands with `sudo microk8s`. For example, `sudo microk8s kubectl ...`.
> To join the group for admin access, avoid using sudo, and other information about `microk8s` setup and usage, review: https://microk8s.io/docs/getting-started.
> If not using `microk8s`, you can use `kubectl` directly. For example, `kubectl get pod`.

```bash
# Export NGC_API_KEY

export NGC_API_KEY=${YOUR_NGC_API_KEY}

# Create credentials for pulling images from NGC (nvcr.io)

sudo microk8s kubectl create secret docker-registry ngc-docker-reg-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password=$NGC_API_KEY

# Configure login information for Neo4j graph database

sudo microk8s kubectl create secret generic graph-db-creds-secret \
    --from-literal=username=neo4j --from-literal=password=password

# Configure login information for ArangoDB graph database
# Note: Need to keep username as root for ArangoDB to work.

sudo microk8s kubectl create secret generic arango-db-creds-secret \
    --from-literal=username=root --from-literal=password=password

# Configure login information for MinIO object storage

sudo microk8s kubectl create secret generic minio-creds-secret \
    --from-literal=access-key=minio --from-literal=secret-key=minio123

# Configure the legacy NGC API key for downloading models from NGC

sudo microk8s kubectl create secret generic ngc-api-key-secret \
--from-literal=NGC_API_KEY=$NGC_API_KEY
```

### OpenAI secret

If using OpenAI, create a secret for the OpenAI API key:

```bash
export OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}

sudo microk8s kubectl create secret generic openai-api-key-secret --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY
```

## Deploy the Helm Chart

To deploy the CA-RAG Blueprint Helm Chart:

```bash
# Make the CA-RAG image

git clone https://github.com/NVIDIA/context-aware-rag.git
cd context-aware-rag
make -C docker build

# import the image to the kubernetes cluster

sudo microk8s enable registry
docker tag vss_ctx_rag-$USER:latest localhost:32000/vss_ctx_rag-$USER:latest
docker push localhost:32000/vss_ctx_rag-$USER:latest

## This will build the CA-RAG image called vss_ctx_rag-<USER>

# Install the Helm Chart

cd helm

## Make sure to replace <USER> with your username.

## Select the appropriate profile for the GPU you are using.

# For H100
sudo microk8s helm install carag-blueprint nvidia-blueprint-carag-1.0.0.tgz \
    --set global.ngcImagePullSecretName=ngc-docker-reg-secret \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.tag=latest \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.tag=latest


# For B200
sudo microk8s helm install carag-blueprint nvidia-blueprint-carag-1.0.0.tgz \
    --set global.ngcImagePullSecretName=ngc-docker-reg-secret \
    --set nim-llm.profile=f17543bf1ee65e4a5c485385016927efe49cbc068a6021573d83eacb32537f76 \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.tag=latest \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.tag=latest

# For H200
sudo microk8s helm install carag-blueprint nvidia-blueprint-carag-1.0.0.tgz \
    --set global.ngcImagePullSecretName=ngc-docker-reg-secret \
    --set nim-llm.profile=99142c13a095af184ae20945a208a81fae8d650ac0fd91747b03148383f882cf \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.tag=latest \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.tag=latest

# For RTX Pro 6000 Blackwell
sudo microk8s helm install carag-blueprint nvidia-blueprint-carag-1.0.0.tgz \
    --set global.ngcImagePullSecretName=ngc-docker-reg-secret \
    --set nim-llm.image.tag=1.13.1 \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.tag=latest \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.tag=latest
```

Wait for all services to be up.
**This can take some time (a few minutes to up to an hour) depending on the setup and configuration.**
Typically, deploying a second time onwards is faster because the models are cached.
Ensure all pods are in Running or Completed STATUS and show 1/1 as READY.
You can monitor the services using the following command:

```bash
sudo watch -n1 microk8s kubectl get pod
```

`watch` refreshes the output every second. Wait for all pods to be in Running or Completed STATUS
and show 1/1 as READY.


To ensure the CA-RAG ingestion and retrieval is ready and accessible, check logs for deployment using command:

```bash
  ## Check the logs for the ingestion
  sudo microk8s kubectl logs -l app.kubernetes.io/name=carag-ingestion

  ## Check the logs for the retrieval service
  sudo microk8s kubectl logs -l app.kubernetes.io/name=carag-retrieval

```

Verify that the following logs are present and that you do not observe errors:


```bash

   Application startup complete.
   Uvicorn running on http://0.0.0.0:8000

```

If a lot of time has passed since CA-RAG started, ``kubectl logs`` might have cleared older logs. In this case look for:

```bash

   INFO:     10.78.15.132:48016 - "GET /health/ready HTTP/1.1" 200 OK
   INFO:     10.78.15.132:50386 - "GET /health/ready HTTP/1.1" 200 OK
   INFO:     10.78.15.132:50388 - "GET /health/live HTTP/1.1" 200 OK
```

## Querying the deployment

Check the NodePort for the carag-ingestion-service and carag-retrieval-service.

```bash
sudo microk8s kubectl get svc | grep carag
carag-ingestion-service                                     NodePort    10.152.183.182   <none>        8000:32657/TCP       4m6s
carag-retrieval-service                                     NodePort    10.152.183.98    <none>        8000:30647/TCP       4m6s
```

Here for example, our carag-ingestion-service is running on port 32657 and carag-retrieval-service is running on port 30647.

### Querying

For querying the service, refer to the [Usage Guide](../../guides/usage/index.md).

Note: Please do not use the `base_url` parameter in the `/chat/completions` endpoint since the service will automatically use the appropriate base URL based on the NIM deployed.




## Default Deployment Topology and Models in Use

The default deployment topology is as follows.

This is the topology that you observe when checking deployment status using `sudo microk8s kubectl get pod`:

| Microservice/Pod | Description | Default #GPUs Allocated |
|------------------|-------------|-------------------------|
| carag-blueprint-0 | The NIM LLM (llama-3.1). | 4 |
| nemo-embedding-embedding-deployment | NeMo Embedding model used in Retrieval Pipeline | 1 |
| nemo-rerank-ranking-deployment | NeMo Reranking model used in Retrieval Pipeline | 1 |
| carag-ingestion-deployment | CA-RAG Ingestion Pipeline | 1 |
| carag-retrieval-deployment | CA-RAG Retrieval Pipeline | 1 |
| etcd, milvus, neo4j, minio, arango-db, elastic-search | Various databases, data stores and supporting services | N/A |



## Configuring GPU Allocation


The default Helm chart deployment topology is configured for 8xGPUs with
each GPU being used by a single service.

To customize the default Helm deployment for various GPU configurations,
modify the ``NVIDIA_VISIBLE_DEVICES`` environment variable for each of the services in the ``overrides.yaml`` file shown below.
Additionally, ``nvidia.com/gpu: 0`` must be set to disable GPU allocation by the GPU operator.

For example, we can deploy reranker and embedding on the same GPU if GPU is 80GB or more.

ca-rag-ingestion and ca-rag-retrieval services do not use much VRAM so we can use the same GPU as well.

Example of 3xH100 deployment overrides which deploys

| Microservice/Pod | Description | GPU Device(s) |
|------------------|-------------|-------------------------|
| carag-blueprint-0 | The NIM LLM (llama-3.1). | 0,1 |
| nemo-embedding-embedding-deployment | NeMo Embedding model used in Retrieval Pipeline | 2 |
| nemo-rerank-ranking-deployment | NeMo Reranking model used in Retrieval Pipeline | 2 |
| carag-ingestion-deployment | CA-RAG Ingestion Pipeline | 2 |
| carag-retrieval-deployment | CA-RAG Retrieval Pipeline | 2 |
| etcd, milvus, neo4j, minio, arango-db, elastic-search | Various databases, data stores and supporting services | N/A |

```yaml
nim-llm:
  env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "0,1"
    - name: NIM_MAX_MODEL_LEN
      value: "128000"
  resources:
    limits:
      nvidia.com/gpu: 0    # no limit

carag-ingestion:
  applicationSpecs:
    deployment:
      containers:
        carag-ingestion-container:
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "2"
          resources:
            limits:
              nvidia.com/gpu: 0    # no limit

carag-retrieval:
  applicationSpecs:
    deployment:
      containers:
        carag-retrieval-container:
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "2"
          resources:
            limits:
              nvidia.com/gpu: 0    # no limit

nemo-embedding:
  applicationSpecs:
    embedding-deployment:
      containers:
        embedding-container:
          env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: '2'
  resources:
    limits:
      nvidia.com/gpu: 0    # no limit

nemo-rerank:
  applicationSpecs:
    ranking-deployment:
      containers:
        ranking-container:
          env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: '2'
  resources:
    limits:
      nvidia.com/gpu: 0    # no limit
```

To run the deployment with the overrides, add `-f overrides.yaml` to the command. Also make sure to not set the nim-llm image profile. For example, if using H200, do not set `nim-llm.profile` to `99142c13a095af184ae20945a208a81fae8d650ac0fd91747b03148383f882cf`. It will auto select the appropriate profile based on the number of GPUs.

```bash
sudo microk8s helm install carag-blueprint nvidia-blueprint-carag-1.0.0.tgz \
    --set global.ngcImagePullSecretName=ngc-docker-reg-secret \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.tag=latest \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.tag=latest \
    -f overrides.yaml
```

## Configuring CA-RAG configs

To configure CA-RAG configs, we can edit the container config in the `overrides.yaml` file.

For example, we can configure the CA-RAG configs to use arango db and graph retrieval.

```yaml
carag-ingestion:
  configs:
    ingestion_config.yaml:
      functions:
        ingestion_function:
          tools:
            db: arango_db
      tools:
        arango_db:
          type: arango
          params:
            host: ${ARANGO_DB_HOST}
            port: ${ARANGO_DB_PORT}
            username: ${ARANGO_DB_USERNAME}
            password: ${ARANGO_DB_PASSWORD}
          tools:
            embedding: nvidia_embedding

carag-retrieval:
  configs:
    retrieval_config.yaml:
      functions:
        retriever_function:
          tools:
            db: arango_db
      tools:
        arango_db:
          type: arango
          params:
            host: ${ARANGO_DB_HOST}
            port: ${ARANGO_DB_PORT}
            username: ${ARANGO_DB_USERNAME}
            password: ${ARANGO_DB_PASSWORD}
          tools:
            embedding: nvidia_embedding

```

If you are only modifying the CA-RAG configs and not GPU topology, you should add the appropriate image profile to the nim-llm. For example, if using H200, set `nim-llm.profile` to `99142c13a095af184ae20945a208a81fae8d650ac0fd91747b03148383f882cf`.

```bash
--set nim-llm.profile=99142c13a095af184ae20945a208a81fae8d650ac0fd91747b03148383f882cf
```

To run the deployment with the overrides, add `-f overrides.yaml` to the command.

```bash
sudo microk8s helm install carag-blueprint nvidia-blueprint-carag-1.0.0.tgz \
    --set global.ngcImagePullSecretName=ngc-docker-reg-secret \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-ingestion.applicationSpecs.deployment.containers.carag-ingestion-container.image.tag=latest \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.repository=localhost:32000/vss_ctx_rag-$USER \
    --set carag-retrieval.applicationSpecs.deployment.containers.carag-retrieval-container.image.tag=latest \
    -f overrides.yaml
```



## Uninstalling the Deployment

To uninstall the deployment, run the following command:

```bash
sudo microk8s helm uninstall carag-blueprint
```
